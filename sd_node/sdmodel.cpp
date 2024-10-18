/*    ModelLoader for Maodot    */
/*       By SleeeepyZhou        */


#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"

#include "util.h"
#include "stb_image_write.h"

#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

#include "sdmodel.h"

SDGGML::SDGGML(int n_threads, 
            rng_type_t rng_type, 
            ggml_backend_t backend,
            SDModel *receiver, const StringName &method): 
                                        n_threads(n_threads),
                                        backend(backend),
                                        receiver(receiver),
                                        method(method) {
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
    }
SDGGML::~SDGGML() {
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

/* Helper */
void SDGGML::printlog(String out_log) {
    if (receiver) {
        receiver->call(method, out_log);
    }
}

bool SDGGML::is_using_v_parameterization_for_sd2(ggml_context* work_ctx) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);
        int64_t t0              = ggml_time_ms();
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model->compute(n_threads, x_t, timesteps, c, NULL, NULL, NULL, -1, {}, 0.f, &out);
        diffusion_model->free_compute_buffer();

        double result = 0.f;
        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }
void calculate_alphas_cumprod(float *alphas_cumprod, float linear_start, 
                                                   float linear_end, int timesteps)  {
        float ls_sqrt = sqrtf(linear_start);
        float le_sqrt = sqrtf(linear_end);
        float amount  = le_sqrt - ls_sqrt;
        float product = 1.0f;
        for (int i = 0; i < timesteps; i++) {
            float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
            product *= 1.0f - powf(beta, 2.0f);
            alphas_cumprod[i] = product;
        }
    }

/* Model */
bool SDGGML::load_from_file(String str_model_path, 
                            ggml_type wtype, 
                            bool clip_on_cpu, 
                            String str_clip_l_path, 
                            String str_t5xxl_path, 
                            bool control_net_cpu, 
                            String str_control_net_path, 
                            String str_embeddings_path, 
                            String str_diffusion_model_path, 
                            String str_id_embeddings_path, 
                            Scheduler schedule, 
                            bool vae_on_cpu, 
                            bool vae_only_decode, 
                            String str_vae_path, 
                            String str_taesd_path, 
                            bool vae_tiling_) {
        /*        Function      */
        bool loadclip      = !str_clip_l_path.is_empty();
        bool loaddiffusion = !str_model_path.is_empty() || !str_diffusion_model_path.is_empty();
        bool loadvae       = !str_vae_path.is_empty() || !str_taesd_path.is_empty();
        bool usecontrolnet = !str_control_net_path.is_empty();
        use_tiny_autoencoder = !str_taesd_path.is_empty();

        /*     gdscript -> C    */
        if (clip_on_cpu && !ggml_backend_is_cpu(clip_backend)) {
            clip_backend = ggml_backend_cpu_init();
        } else {
            clip_backend = backend;
        }
        if (vae_on_cpu && !ggml_backend_is_cpu(vae_backend)) {
            vae_backend = ggml_backend_cpu_init();
        } else {
            vae_backend = backend;
        }
        if (control_net_cpu && !ggml_backend_is_cpu(control_net_backend)) {
            control_net_backend = ggml_backend_cpu_init();
        } else {
            control_net_backend = backend;
        }
        const std::string model_path(str_model_path.utf8().get_data());
        const std::string clip_l_path(str_clip_l_path.utf8().get_data());
        const std::string t5xxl_path(str_t5xxl_path.utf8().get_data());
        const std::string embeddings_path(str_embeddings_path.utf8().get_data());
        const std::string control_net_path(str_control_net_path.utf8().get_data());
        const std::string diffusion_model_path(str_diffusion_model_path.utf8().get_data());
        const std::string id_embeddings_path(str_id_embeddings_path.utf8().get_data());
        const std::string vae_path(str_vae_path.utf8().get_data());
        const std::string taesd_path(str_taesd_path.utf8().get_data());
        
        vae_decode_only = vae_only_decode;
        vae_tiling = vae_tiling_;
    
        /*    Loader in file    */
        ModelLoader model_loader;

        if (!str_model_path.is_empty()) {
            printlog(vformat("Loading model from '%s'", str_model_path));
            if (!model_loader.init_from_file(model_path)) {
                ERR_PRINT(vformat("Model loader failed: '%s'", str_model_path));
            }
        }
        if (loadclip) {
            printlog(vformat("Loading clip_l from '%s'", str_clip_l_path));
            if (!model_loader.init_from_file(clip_l_path, "text_encoders.clip_l.")) {
                ERR_PRINT(vformat("Loading clip_l '%s' failed", str_clip_l_path));
            }
        }
        if (!str_t5xxl_path.is_empty()) {
            printlog(vformat("loading t5xxl from '%s'", str_t5xxl_path));
            if (!model_loader.init_from_file(t5xxl_path, "text_encoders.t5xxl.")) {
                ERR_PRINT(vformat("loading t5xxl from '%s' failed", str_t5xxl_path));
            }
        }
        if (!str_diffusion_model_path.is_empty()) {
            printlog(vformat("loading diffusion model from '%s'", str_diffusion_model_path));
            if (!model_loader.init_from_file(diffusion_model_path, "model.diffusion_model.")) {
                ERR_PRINT(vformat("loading diffusion model from '%s' failed", str_diffusion_model_path));
            }
        }
        if (!str_vae_path.is_empty()) {
            printlog(vformat("loading vae from '%s'", str_vae_path));
            if (!model_loader.init_from_file(vae_path, "vae.")) {
                ERR_PRINT(vformat("loading vae '%s' failed", str_vae_path));
            }
        }
        if (!str_id_embeddings_path.is_empty()) {
            printlog(vformat("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", str_id_embeddings_path));
            if (!model_loader.init_from_file(id_embeddings_path, "pmid.")) {
                ERR_PRINT(vformat("loading stacked ID embedding from '%s' failed", str_id_embeddings_path));
                stacked_id = false;
            } else {
                stacked_id = true;
            }
        }

        /*      Get version     */
        if (loaddiffusion) {
            version = model_loader.get_sd_version();
            if (version == VERSION_COUNT) {
                ERR_PRINT(vformat("Get sd version from file failed."));
                return false;
            }
            if (version == VERSION_SDXL && !loadvae) {
                WARN_PRINT(vformat(
                    "### It looks like you are using SDXL model. ###"
                    "If you find that the generated images are completely black, "
                    "try specifying SDXL VAE FP16 Fix. You can find it here: "))
                print_line_rich("[url=https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors]sdxl_vae.safetensors[/url]");
            }
            printlog(vformat("Version: %s ", model_version_to_str[version]));
        } else if (version == VERSION_COUNT) {
            ERR_PRINT(vformat("It looks like there no diffusion model."))
        }
        bool use_t5xxl = false;
        if (version == VERSION_SD3_2B || version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
            use_t5xxl = true;
        }
        
        /*    Get weight type   */
        printlog(vformat("Get model weight type..."));
        if (wtype == GGML_TYPE_COUNT) {
            if (!str_model_path.is_empty()) {
                model_wtype = model_loader.get_sd_wtype();
                if (model_wtype == GGML_TYPE_COUNT) {
                    model_wtype = GGML_TYPE_F32;
                    ERR_PRINT(vformat("Can not get mode wtype from weight, use f32"));
                }
            }
            if (!str_model_path.is_empty() || loadclip) {
                conditioner_wtype = model_loader.get_conditioner_wtype();}
            if (loaddiffusion) {
                diffusion_model_wtype = model_loader.get_diffusion_model_wtype();}
            if (!str_model_path.is_empty() || !str_vae_path.is_empty()) {
                vae_wtype = model_loader.get_vae_wtype();}
        } else {
            model_wtype           = wtype;
            conditioner_wtype     = wtype;
            diffusion_model_wtype = wtype;
            vae_wtype             = wtype;
        }
        if (version == VERSION_SDXL) {
            vae_wtype = GGML_TYPE_F32;
        }
        printlog(vformat("Model weight type:           %s", ggml_type_name(model_wtype)));
        printlog(vformat("Conditioner weight type:     %s", ggml_type_name(conditioner_wtype)));
        printlog(vformat("Diffusion model weight type: %s", ggml_type_name(diffusion_model_wtype)));
        printlog(vformat("VAE weight type:             %s", ggml_type_name(vae_wtype)));

        printlog(vformat("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor)));
        
        if (!ggml_backend_is_cpu(clip_backend) && use_t5xxl && conditioner_wtype != GGML_TYPE_F32) {
            printlog(vformat("CLIP: Using CPU"));
            clip_backend = ggml_backend_cpu_init();
        }

        /*     scale_factor     */
        if (loaddiffusion) {
            if (version == VERSION_SDXL) {
                scale_factor = 0.13025f;
            } else if (version == VERSION_SD3_2B) {
                scale_factor = 1.5305f;
            } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
                scale_factor = 0.3611;
            } else {
                scale_factor = 0.18215f;
            }
        }

        /*      Creat Model     */
        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, conditioner_wtype);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors);

            diffusion_model = std::make_shared<UNetModel>(backend, diffusion_model_wtype, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, vae_wtype, vae_decode_only, true, version);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
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

            if (use_tiny_autoencoder) {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend, vae_wtype, vae_decode_only);
                if (!tae_first_stage->load_from_file(taesd_path)) {
                    return false;
                }
            } else {
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_wtype, vae_decode_only, false, version);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            }

            if (usecontrolnet) {
                control_net = std::make_shared<ControlNet>(control_net_backend, diffusion_model_wtype, version);
                if (!control_net->load_from_file(control_net_path)) {
                    return false;
                }
            }

            if (stacked_id) {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(clip_backend, model_wtype, version);
                if (!pmid_model->alloc_params_buffer()) {
                    ERR_PRINT(" pmid model params buffer allocation failed");
                    return false;
                }
                pmid_lora = std::make_shared<LoraModel>(backend, model_wtype, id_embeddings_path, "");
                if (!pmid_lora->load_from_file(true)) {
                    ERR_PRINT(vformat("load photomaker lora tensors from %s failed", str_id_embeddings_path));
                    return false;
                }
                pmid_model->get_param_tensors(tensors, "pmid");
                //    pmid_model.init_params(GGML_TYPE_F32);
                //    pmid_model.map_by_name(tensors, "pmid.");
            }
        }

        /*     load weights     */
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        printlog(vformat("loading weights"));
        int64_t t0 = ggml_time_ms();
        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        } else if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        } 
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            ERR_PRINT(vformat("load tensors from model loader failed"));
            ggml_free(ctx);
            return false;
        }

        /*       mem_size       */
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
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }

            size_t control_net_params_mem_size = 0;
            if (control_net) {
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
            printlog(vformat(
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
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM"));
        }

        int64_t t1 = ggml_time_ms();
        printlog(vformat("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1 - t0) * 1.0f / 1000));

        if (loaddiffusion) {
        /*       denoiser       */
        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = (version == VERSION_SD2 && is_using_v_parameterization_for_sd2(ctx));
        if (version == VERSION_SD3_2B) {
            printlog(vformat("running in FLOW mode"));
            denoiser = std::make_shared<DiscreteFlowDenoiser>();
        } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
            printlog(vformat("running in Flux FLOW mode"));
            float shift = 1.15f;
            if (version == VERSION_FLUX_SCHNELL) {
                shift = 1.0f;  // TODO: validate
            }
            denoiser = std::make_shared<FluxFlowDenoiser>(shift);
        } else if (is_using_v_parameterization || version == VERSION_SVD) {
            printlog(vformat("running in v-prediction mode"));
            denoiser = std::make_shared<CompVisVDenoiser>();
        } else {
            printlog(vformat("running in eps-prediction mode"));
        }

        /*      scheduler       */
        if (schedule != DEFAULT) {
            switch (schedule) {
                case DISCRETE:
                    printlog(vformat("running with discrete schedule"));
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    printlog(vformat("running with Karras schedule"));
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case EXPONENTIAL:
                    printlog(vformat("running exponential schedule"));
                    denoiser->schedule = std::make_shared<ExponentialSchedule>();
                    break;
                case AYS:
                    printlog(vformat("Running with Align-Your-Steps schedule"));
                    denoiser->schedule          = std::make_shared<AYSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case GITS:
                    printlog(vformat("Running with GITS schedule"));
                    denoiser->schedule          = std::make_shared<GITSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case DEFAULT:
                    break;
                default:
                    printlog(vformat("Unknown schedule %i", schedule));
                    abort();
            }
        }
        
        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            for (int i = 0; i < TIMESTEPS; i++) {
                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }
        }

        printlog(vformat("Finished loaded file"));
        ggml_free(ctx);
        return true;
    }

void SDGGML::apply_lora(const std::string &lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            ERR_PRINT(vformat("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str()));
            return;
        }
        LoraModel lora(backend, model_wtype, file_path);
        if (!lora.load_from_file()) {
            ERR_PRINT(vformat("load lora tensors from %s failed", file_path.c_str()));
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        printlog(vformat("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000));
    }
void SDGGML::apply_loras(const std::unordered_map<std::string, float> &lora_state) {
        if (lora_state.size() > 0 && model_wtype != GGML_TYPE_F16 && model_wtype != GGML_TYPE_F32) {
            ERR_PRINT(vformat("In quantized models when applying LoRA, the images have poor quality."));
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

        printlog(vformat("Attempting to apply %lu LoRAs", lora_state.size()));

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

ggml_tensor *SDGGML::id_encoder(ggml_context *work_ctx, ggml_tensor *init_img, ggml_tensor *prompts_embeds, std::vector<bool> &class_tokens_mask) {
        ggml_tensor* res = NULL;
        pmid_model->compute(n_threads, init_img, prompts_embeds, class_tokens_mask, &res, work_ctx);
        return res;
    }
SDCondition SDGGML::get_svd_condition(ggml_context *work_ctx, sd_image_t init_image, 
                                                    int width, int height, int fps, int motion_bucket_id, 
                                                    float augmentation_level, bool force_zero_embeddings) {
        // c_crossattn
        int64_t t0                      = ggml_time_ms();
        struct ggml_tensor* c_crossattn = NULL;
        {
            if (force_zero_embeddings) {
                c_crossattn = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, clip_vision->vision_model.projection_dim);
                ggml_set_f32(c_crossattn, 0.f);
            } else {
                sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size);
                free(image.data);
                image.data = NULL;

                ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_tensor(resized_image.data, pixel_values, false);
                free(resized_image.data);
                resized_image.data = NULL;

                // print_ggml_tensor(pixel_values);
                clip_vision->compute(n_threads, pixel_values, &c_crossattn, work_ctx);
                // print_ggml_tensor(c_crossattn);
            }
        }

        // c_concat
        struct ggml_tensor* c_concat = NULL;
        {
            if (force_zero_embeddings) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = NULL;
                    sd_image_f32_to_tensor(resized_image.data, init_img, false);
                    free(resized_image.data);
                    resized_image.data = NULL;
                } else {
                    sd_image_to_tensor(init_image.data, init_img);
                }
                if (augmentation_level > 0.f) {
                    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_tensor_set_f32_randn(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_tensor_scale(noise, augmentation_level);
                    ggml_tensor_add(init_img, noise);
                }
                ggml_tensor* moments = encode_first_stage(work_ctx, init_img);
                c_concat             = get_first_stage_encoding(work_ctx, moments);
            }
        }

        // y
        struct ggml_tensor* y = NULL;
        {
            y                            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->get_adm_in_channels());
            int out_dim                  = 256;
            int fps_id                   = fps - 1;
            std::vector<float> timesteps = {(float)fps_id, (float)motion_bucket_id, augmentation_level};
            set_timestep_embedding(timesteps, y, out_dim);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing svd condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {c_crossattn, y, c_concat};
    }

ggml_tensor *SDGGML::sample(ggml_context *work_ctx, ggml_tensor *init_latent, 
                                        ggml_tensor *noise, SDCondition cond, SDCondition uncond, 
                                        ggml_tensor *control_hint, float control_strength, 
                                        float min_cfg, float cfg_scale, float guidance, 
                                        Scheduler method, const std::vector<float> &sigmas, 
                                        int start_merge_step, SDCondition id_cond) {
        size_t steps = sigmas.size() - 1;
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, noise);

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = NULL;
        if (cfg_scale != 1.0) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }

        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);
        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            /*
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            */
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
            
            // uncond
            float* negative_data = NULL;
            if (cfg_scale != 1.0) {
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
                if (cfg_scale != 1.0) {
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
            /*
            if (step > 0) {
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                // printlog(vformat("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
            */

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

ggml_tensor *SDGGML::get_first_stage_encoding(ggml_context *work_ctx, ggml_tensor *moments) {
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
ggml_tensor *SDGGML::compute_first_stage(ggml_context *work_ctx, ggml_tensor *x, bool decode){
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
ggml_tensor *SDGGML::encode_first_stage(ggml_context *work_ctx, ggml_tensor *x) {
        return compute_first_stage(work_ctx, x, false);
    }
ggml_tensor *SDGGML::decode_first_stage(ggml_context *work_ctx, ggml_tensor *x) {
        return compute_first_stage(work_ctx, x, true);
    }

void SDGGML::_bind_methods() {
}


// SDModel
SDModel::SDModel() {
}
SDModel::~SDModel() {
}

/* Helper */
Array SDModel::get_vk_devices_idx() const {
    return DeviceInfo::getInstance().get_devices_idx();
}
ggml_backend_t SDModel::set_device(int device_index, bool use_cpu) {
    ggml_backend_t backend;
    if (!use_cpu) {
/*
#ifdef SD_USE_CUBLAS
    printlog(vformat("Using CUDA backend"));
    backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
    printlog(vformat("Using Metal backend"));
    ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
    backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_SYCL
    printlog(vformat("Using SYCL backend"));
    backend = ggml_backend_sycl_init(0);
#endif
#ifdef SD_USE_VULKAN
*/
    printlog(vformat("Using Vulkan"));
    Array vk_devices_idx = get_vk_devices_idx();
    if (!vk_devices_idx.is_empty()) {
        size_t set_device = vk_devices_idx[0];
        if (device_index >= 0 && vk_devices_idx.has(device_index)) {
            set_device = device_index;
        } 
        backend = ggml_backend_vk_init(set_device);
        printlog(vformat("Using Vulkan device : %d", set_device));
    }
    if (!backend) {
        WARN_PRINT(vformat("Failed to initialize Vulkan backend"));
    }
/*
#endif
*/
    }
    if (!backend || use_cpu) {
        WARN_PRINT(vformat("Using CPU backend"));
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

String SDModel::get_model_path() const {
	return model_path;
}
SDVersion SDModel::get_version() const {
	return version;
}
Scheduler SDModel::get_schedule() const {
	return schedule;
}

/* Model */
void SDModel::load_model(String str_model_path, int device_index, Scheduler schedule, 
                        bool use_cpu, bool vae_on_cpu, bool clip_on_cpu) {
    Array result;
    if (sd != nullptr) {sd = nullptr;}
    ggml_backend_t backend = set_device(device_index, use_cpu);
    if (!backend) {
        result.push_back(false);
        result.push_back(vformat("Failed to initialize backend."));
        emit_signal(SNAME("load_log"), result);
        return;
    }
	sd = new SDGGML(n_threads, 
                STD_DEFAULT_RNG, 
                backend,
                this, "printlog");
    if (!sd.is_valid()) {
        result.push_back(false);
        result.push_back(vformat("Failed to create SD."));
        emit_signal(SNAME("load_log"), result);
        return;
    }
    if (!sd->load_from_file(str_model_path, GGML_TYPE_COUNT, 
                            clip_on_cpu, "","",false,"","","","",
                            schedule,vae_on_cpu,false,"","",false)) {
        sd = nullptr;
        result.push_back(false);
        result.push_back(vformat("Failed to load model."));
        emit_signal(SNAME("load_log"), result);
        return;
    }
    model_path = str_model_path;
    version = sd->version;
    schedule = schedule;
    result.push_back(true);
    result.push_back(vformat("Loading completed. Version: %s.", model_version_to_str[version]));
    emit_signal(SNAME("load_log"), result);
}

/* Inference */
void SDModel::ksample(Latent init_latent, SDCond cond_res, 
                      int steps, float CFG, float denoise, SamplerName sampler_name, 
                      int seed) {
    int64_t t1 = ggml_time_ms();
    /* gdscript -> C */
    if (!sd.is_valid()) {
        ERR_PRINT(vformat("No model is loaded."));
        return;
    }
    Array latent_info = latent.get_latent_info();
    if (!latent_info[0]) { 
        ERR_PRINT(vformat("No latent."));
        return;
    }
    int width = latent_info[1];
    int height = latent_info[2];
    int batch_count = latent_info[3];
    struct ggml_context* work_ctx = latent.get_work_ctx();

    SDCondition cond = cond_res.get_cond();
    SDCondition uncond = cond_res.get_uncond();

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    
    std::vector<float> sigmas = sd->denoiser->get_sigmas(steps);
    int sample_steps = sigmas.size() - 1;

    /* Sample */

    // latents
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    if (sd->version == VERSION_SD3_2B) {
        C = 16;
    } else if (sd->version == VERSION_FLUX_DEV || sd->version == VERSION_FLUX_SCHNELL) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;

    // sample
    printlog(vformat("Sampling using %s method", sampling_methods_str[sampler_name]));
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        printlog(vformat("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed));

        sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd->rng);

        int start_merge_step = -1;
        if (sd->stacked_id) {
            start_merge_step = int(sd->pmid_model->style_strength / 100.f * sample_steps);
            printlog(vformat("PHOTOMAKER: start_merge_step: %d", start_merge_step));
        }

        struct ggml_tensor* x_0 = sd->sample(work_ctx,
                                            x_t,
                                            noise,
                                            cond,
                                            uncond,
                                            NULL,
                                            0.0f,
                                            CFG,
                                            CFG,
                                            0.0f,
                                            sampler_name,
                                            sigmas,
                                            start_merge_step,
                                            NULL);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        printlog(vformat("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000));
        final_latents.push_back(x_0);
    }

    if (sd->free_params_immediately) {
        sd->diffusion_model->free_params_buffer();
    }
    int64_t t2 = ggml_time_ms();
    printlog(vformat("Generating latent completed, taking %.2fs", final_latents.size(), (t2 - t1) * 1.0f / 1000));
    init_latent.take_result_latent(final_latents);
}

/* VAE */
#include "core/io/image.h"

void SDModel::decode(Latent init_latent) {
    Array result;
    if (!init_latent.has_result) {
        ERR_PRINT(vformat("No final latents"));
        result.push_back(false);
        result.push_back(vformat("No final latents"));
        emit_signal(SNAME("vae_log"), result);
        return;
    }
    std::vector<struct ggml_tensor*> final_latents;
    final_latents = init_latent.get_final_latents();
    Array latent_info = latent.get_latent_info();
    int width = latent_info[1];
    int height = latent_info[2];
    int batch_count = latent_info[3];
    struct ggml_context* work_ctx = init_latent.get_work_ctx();

    int64_t t3 = ggml_time_ms();
    // Decode
    printlog(vformat("decoding %zu latents", final_latents.size()));
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1 = ggml_time_ms();
        struct ggml_tensor* img = sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        printlog(vformat("latent decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000));
    }

    int64_t t4 = ggml_time_ms();
    printlog(vformat("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000));
    if (sd->free_params_immediately && !sd->use_tiny_autoencoder) {
        sd->first_stage_model->free_params_buffer();
    }

    // to image
    Array png_images;
    bool no_error = true;
    for (size_t i = 0; i < decoded_images.size(); i++) {
        int len;
        std::string parameters = "";
        unsigned char *png = stbi_write_png_to_mem((const unsigned char *) sd_tensor_to_image(decoded_images[i]), 
                                                0, width, height, 3, 
                                                &len, parameters.c_str());
        if (png == NULL) {
            ERR_PRINT("Png write failed.");
            no_error = false;
            continue;
        }
        Ref<Image> image;
        image.instantiate();
        Vector<uint8_t> byte_vector;
        byte_vector.resize(len); 
        for (int i = 0; i < len; i++) {
            byte_vector.write[i] = png_data[i];
        }
        Error err = image->load_png_from_buffer(byte_vector);
        if (err != OK) {
            ERR_PRINT("Failed to load PNG data into Image.");
            no_error = false;
            continue;
        }
        png_images.push_back(image);
    }
    result.push_back(no_error);
    result.push_back(png_images);
    emit_signal(SNAME("vae_log"), result);
}

void SDModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_vk_devices_idx"), &SDModel::get_vk_devices_idx);
	ClassDB::bind_method(D_METHOD("get_model_path"), &SDModel::get_model_path);
    ClassDB::bind_method(D_METHOD("get_version"), &SDModel::get_version);

    ClassDB::bind_method(D_METHOD("load_model","model_path","device_index","scheduler","use_cpu","vae_on_cpu","clip_on_cpu"), &SDModel::load_model);
    ClassDB::bind_method(D_METHOD("ksample","init_latent","cond_res","steps","CFG","denoise","sampler_name","seed"), &SDModel::ksample);
    ClassDB::bind_method(D_METHOD("decode","init_latent"), &SDModel::decode);

    ADD_SIGNAL(MethodInfo("load_log", PropertyInfo(Variant::ARRAY, "load_info")));
    ADD_SIGNAL(MethodInfo("vae_log", PropertyInfo(Variant::ARRAY, "vae_result")));

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "", PROPERTY_USAGE_READ_ONLY),"","get_model_path");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "version", PROPERTY_HINT_ENUM, "SD1.x, SD2.x, SDXL, SVD, SD3-2B, FLUX-dev, FLUX-schnell", PROPERTY_USAGE_READ_ONLY),"","get_version");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "schedule", PROPERTY_HINT_ENUM, "default, discrete, karras, exponential, ays, gits", PROPERTY_USAGE_READ_ONLY),"","get_schedule");

    BIND_ENUM_CONSTANT(VERSION_SD1);
    BIND_ENUM_CONSTANT(VERSION_SD2);
	BIND_ENUM_CONSTANT(VERSION_SDXL);
    BIND_ENUM_CONSTANT(VERSION_SVD);
    BIND_ENUM_CONSTANT(VERSION_SD3_2B);
    BIND_ENUM_CONSTANT(VERSION_FLUX_DEV);
    BIND_ENUM_CONSTANT(VERSION_FLUX_SCHNELL);
    BIND_ENUM_CONSTANT(VERSION_COUNT);

    BIND_ENUM_CONSTANT(DEFAULT);
    BIND_ENUM_CONSTANT(DISCRETE);
	BIND_ENUM_CONSTANT(KARRAS);
    BIND_ENUM_CONSTANT(EXPONENTIAL);
    BIND_ENUM_CONSTANT(AYS);
    BIND_ENUM_CONSTANT(GITS);
    BIND_ENUM_CONSTANT(N_SCHEDULES);


    BIND_ENUM_CONSTANT(EULER_A);
    BIND_ENUM_CONSTANT(EULER);
	BIND_ENUM_CONSTANT(HEUN);
    BIND_ENUM_CONSTANT(DPM2);
    BIND_ENUM_CONSTANT(DPMPP2S_A);
    BIND_ENUM_CONSTANT(DPMPP2M);
    BIND_ENUM_CONSTANT(DPMPP2Mv2);
    BIND_ENUM_CONSTANT(IPNDM);
    BIND_ENUM_CONSTANT(IPNDM_V);
    BIND_ENUM_CONSTANT(LCM);
    BIND_ENUM_CONSTANT(N_SAMPLE_METHODS);
}