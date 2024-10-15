#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"

#include "stable-diffusion.h"
#include "util.h"

#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
};

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t control_net_backend = NULL;
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type conditioner_wtype        = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;
    ggml_type vae_wtype                = GGML_TYPE_COUNT;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;

    std::string taesd_path;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;
    bool stacked_id           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        rng_type_t rng_type)
        : n_threads(n_threads),
          vae_decode_only(vae_decode_only),
          free_params_immediately(free_params_immediately),
          lora_model_dir(lora_model_dir) {
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
    }

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& clip_l_path,
                        const std::string& t5xxl_path,
                        const std::string& diffusion_model_path,
                        const std::string& vae_path,
                        const std::string control_net_path,
                        const std::string embeddings_path,
                        const std::string id_embeddings_path,
                        const std::string& taesd_path,
                        bool vae_tiling_,
                        ggml_type wtype,
                        schedule_t schedule,
                        bool clip_on_cpu,
                        bool control_net_cpu,
                        bool vae_on_cpu) {
        use_tiny_autoencoder = taesd_path.size() > 0;


        ModelLoader model_loader;



        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, conditioner_wtype);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors);

            diffusion_model = std::make_shared<UNetModel>(backend, diffusion_model_wtype, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, vae_wtype, vae_decode_only, true, version);
            LOG_DEBUG("vae_decode_only %d", vae_decode_only);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
            clip_backend   = backend;
            bool use_t5xxl = false;
            if (version == VERSION_SD3_2B || version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
                use_t5xxl = true;
            }
            if (!ggml_backend_is_cpu(backend) && use_t5xxl && conditioner_wtype != GGML_TYPE_F32) {
                clip_on_cpu = true;
                LOG_INFO("set clip_on_cpu to true");
            }
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (version == VERSION_SD3_2B) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend, conditioner_wtype);
                diffusion_model  = std::make_shared<MMDiTModel>(backend, diffusion_model_wtype, version);
            } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
                cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend, conditioner_wtype);
                diffusion_model  = std::make_shared<FluxModel>(backend, diffusion_model_wtype, version);
            } else {
                cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, conditioner_wtype, embeddings_path, version);
                diffusion_model  = std::make_shared<UNetModel>(backend, diffusion_model_wtype, version);
            }
            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            if (!use_tiny_autoencoder) {
                if (vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_INFO("VAE Autoencoder: Using CPU backend");
                    vae_backend = ggml_backend_cpu_init();
                } else {
                    vae_backend = backend;
                }
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_wtype, vae_decode_only, false, version);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            } else {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend, vae_wtype, vae_decode_only);
            }
            // first_stage_model->get_param_tensors(tensors, "first_stage_model.");

        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);



        // load weights
        LOG_DEBUG("loading weights");

        int64_t t0 = ggml_time_ms();

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);
        if (version == VERSION_SVD) {
            // diffusion_model->test();
            // first_stage_model->test();
            // return false;
        } else {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = 0;
            if (!use_tiny_autoencoder) {
                vae_params_mem_size = first_stage_model->get_params_buffer_size();
            } else {
                if (!tae_first_stage->load_from_file(taesd_path)) {
                    return false;
                }
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(control_net_path)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (stacked_id) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (ggml_backend_is_cpu(control_net_backend)) {
                total_params_ram_size += control_net_params_mem_size;
            } else {
                total_params_vram_size += control_net_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1 - t0) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (version == VERSION_SD2) {
            if (is_using_v_parameterization_for_sd2(ctx)) {
                is_using_v_parameterization = true;
            }
        } else if (version == VERSION_SVD) {
            // TODO: V_PREDICTION_EDM
            is_using_v_parameterization = true;
        }

        if (version == VERSION_SD3_2B) {
            LOG_INFO("running in FLOW mode");
            denoiser = std::make_shared<DiscreteFlowDenoiser>();
        } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
            LOG_INFO("running in Flux FLOW mode");
            float shift = 1.15f;
            if (version == VERSION_FLUX_SCHNELL) {
                shift = 1.0f;  // TODO: validate
            }
            denoiser = std::make_shared<FluxFlowDenoiser>(shift);
        } else if (is_using_v_parameterization) {
            LOG_INFO("running in v-prediction mode");
            denoiser = std::make_shared<CompVisVDenoiser>();
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (schedule != DEFAULT) {
        }

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            for (int i = 0; i < TIMESTEPS; i++) {
                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }

        LOG_DEBUG("finished loaded file");
        ggml_free(ctx);
        return true;
    }

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            LOG_WARN("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str());
            return;
        }
        LoraModel lora(backend, model_wtype, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        if (lora_state.size() > 0 && model_wtype != GGML_TYPE_F16 && model_wtype != GGML_TYPE_F32) {
            LOG_WARN("In quantized models when applying LoRA, the images have poor quality.");
        }
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;

            if (curr_lora_state.find(lora_name) != curr_lora_state.end()) {
                float curr_multiplier = curr_lora_state[lora_name];
                float multiplier_diff = multiplier - curr_multiplier;
                if (multiplier_diff != 0.f) {
                    lora_state_diff[lora_name] = multiplier_diff;
                }
            } else {
                lora_state_diff[lora_name] = multiplier;
            }
        }

        LOG_INFO("Attempting to apply %lu LoRAs", lora_state.size());

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        float min_cfg,
                        float cfg_scale,
                        float guidance,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        SDCondition id_cond) {
        size_t steps = sigmas.size() - 1;
        // noise = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(noise);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, noise);

        bool has_unconditioned = cfg_scale != 1.0 && uncond.c_crossattn != NULL;

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = NULL;
        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec(x->ne[3], t);  // [N, ]
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(x->ne[3], guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            if (control_hint != NULL) {
                control_net->compute(n_threads, noised_input, control_hint, timesteps, cond.c_crossattn, cond.c_vector);
                controls = control_net->controls;
                // print_ggml_tensor(controls[12]);
                // GGML_ASSERT(0);
            }

            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         cond.c_crossattn,
                                         cond.c_concat,
                                         cond.c_vector,
                                         guidance_tensor,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            } else {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         id_cond.c_crossattn,
                                         cond.c_concat,
                                         id_cond.c_vector,
                                         guidance_tensor,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            }

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                if (control_hint != NULL) {
                    control_net->compute(n_threads, noised_input, control_hint, timesteps, uncond.c_crossattn, uncond.c_vector);
                    controls = control_net->controls;
                }
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         uncond.c_crossattn,
                                         uncond.c_concat,
                                         uncond.c_vector,
                                         guidance_tensor,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_uncond);
                negative_data = (float*)out_uncond->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);
            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    int64_t ne3 = out_cond->ne[3];
                    if (min_cfg != cfg_scale && ne3 != 1) {
                        int64_t i3  = i / out_cond->ne[0] * out_cond->ne[1] * out_cond->ne[2];
                        float scale = min_cfg + (cfg_scale - min_cfg) * (i3 * 1.0f / ne3);
                    } else {
                        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                    }
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }
            int64_t t1 = ggml_time_us();
            if (step > 0) {
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
            return denoised;
        };

        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng);

        x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        diffusion_model->free_compute_buffer();
        return x;
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(work_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t C = 8;
        if (use_tiny_autoencoder) {
            C = 4;
        } else {
            if (version == VERSION_SD3_2B) {
                C = 32;
            } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
                C = 32;
            }
        }
        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : C,
                                                 x->ne[3]);  // channels
        int64_t t0          = ggml_time_ms();
        if (!use_tiny_autoencoder) {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 32x32 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    first_stage_model->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model->compute(n_threads, x, decode, &result);
            }
            first_stage_model->free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    tae_first_stage->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage->compute(n_threads, x, decode, &result);
            }
            tae_first_stage->free_compute_buffer();
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }
};

/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};

sd_ctx_t* new_sd_ctx(const char* model_path_c_str,
                     const char* clip_l_path_c_str,
                     const char* t5xxl_path_c_str,
                     const char* diffusion_model_path_c_str,
                     const char* vae_path_c_str,
                     const char* taesd_path_c_str,
                     const char* control_net_path_c_str,
                     const char* lora_model_dir_c_str,
                     const char* embed_dir_c_str,
                     const char* id_embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum sd_type_t wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_clip_on_cpu,
                     bool keep_control_net_cpu,
                     bool keep_vae_on_cpu) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string model_path(model_path_c_str);
    std::string clip_l_path(clip_l_path_c_str);
    std::string t5xxl_path(t5xxl_path_c_str);
    std::string diffusion_model_path(diffusion_model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string id_embd_path(id_embed_dir_c_str);
    std::string lora_model_dir(lora_model_dir_c_str);

    sd_ctx->sd = new StableDiffusionGGML(n_threads,
                                         vae_decode_only,
                                         free_params_immediately,
                                         lora_model_dir,
                                         rng_type);
    if (sd_ctx->sd == NULL) {
        return NULL;
    }

    if (!sd_ctx->sd->load_from_file(model_path,
                                    clip_l_path,
                                    t5xxl_path_c_str,
                                    diffusion_model_path,
                                    vae_path,
                                    control_net_path,
                                    embd_path,
                                    id_embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_clip_on_cpu,
                                    keep_control_net_cpu,
                                    keep_vae_on_cpu)) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
        free(sd_ctx);
        return NULL;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != NULL) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,
                           ggml_tensor* init_latent,
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_ratio,
                           bool normalize_input,
                           std::string input_id_images_path) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    // for (auto v : sigmas) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    // Get learned condition
    t0               = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_SDXL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    if (sd_ctx->sd->version == VERSION_SD3_2B) {
        C = 16;
    } else if (sd_ctx->sd->version == VERSION_FLUX_DEV || sd_ctx->sd->version == VERSION_FLUX_SCHNELL) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     image_hint,
                                                     control_strength,
                                                     cfg_scale,
                                                     cfg_scale,
                                                     guidance,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     id_cond);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(decoded_images[i]);
    }
    ggml_free(work_ctx);

    return result_images;
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_ctx->sd->version == VERSION_SD3_2B) {
        params.mem_size *= 3;
    }
    if (sd_ctx->sd->version == VERSION_FLUX_DEV || sd_ctx->sd->version == VERSION_FLUX_SCHNELL) {
        params.mem_size *= 4;
    }
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float);
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    int C = 4;
    if (sd_ctx->sd->version == VERSION_SD3_2B) {
        C = 16;
    } else if (sd_ctx->sd->version == VERSION_FLUX_DEV || sd_ctx->sd->version == VERSION_FLUX_SCHNELL) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;
    ggml_tensor* init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    if (sd_ctx->sd->version == VERSION_SD3_2B) {
        ggml_set_f32(init_latent, 0.0609f);
    } else if (sd_ctx->sd->version == VERSION_FLUX_DEV || sd_ctx->sd->version == VERSION_FLUX_SCHNELL) {
        ggml_set_f32(init_latent, 0.1159f);
    } else {
        ggml_set_f32(init_latent, 0.f);
    }

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t1 = ggml_time_ms();

    LOG_INFO("txt2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}

sd_image_t* img2img(sd_ctx_t* sd_ctx,
                    sd_image_t init_image,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    int width,
                    int height,
                    sample_method_t sample_method,
                    int sample_steps,
                    float strength,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("img2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_ctx->sd->version == VERSION_SD3_2B) {
        params.mem_size *= 2;
    }
    if (sd_ctx->sd->version == VERSION_FLUX_DEV || sd_ctx->sd->version == VERSION_FLUX_SCHNELL) {
        params.mem_size *= 3;
    }
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    sd_image_to_tensor(init_image.data, init_img);
    ggml_tensor* init_latent = NULL;
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    }
    print_ggml_tensor(init_latent, true);
    size_t t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               width,
                                               height,
                                               sample_method,
                                               sigma_sched,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t2 = ggml_time_ms();

    LOG_INFO("img2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}