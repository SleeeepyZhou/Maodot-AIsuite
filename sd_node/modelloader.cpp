/*    ModelLoader for Maodot    */
/*       By SleeeepyZhou        */

#include "ggml_extend.hpp"
#include "model.h"

#include "conditioner.hpp"
#include "diffusion_model.hpp"
#include "denoiser.hpp"
#include "esrgan.hpp"
#include "vae.hpp"
#include "tae.hpp"

#include "stablediffusion.h"
#include "modelloader.h"

/*

Res
Backend
SDModel
    CLIP
    Diffusion
    VAE/TinyAE

Node
SDModelLoader
    in  model_path
    out CLIP
        Diffusion
        VAE

*/

// Backend
Backend::Backend() {
}
Backend::Backend(int _index, bool _cpu) {
    usecpu(_cpu);
    set_device(_index);
}
Backend::~Backend() {
    if (backend) {
        ggml_backend_free(backend);
    }
}

ggml_backend_t Backend::get_backend() const {
	return backend;
}

Array Backend::get_vk_devices_idx() const {
    return DeviceInfo::getInstance().get_devices_idx();
}
void Backend::set_device(int device_index) {
    if (backend) {
        ggml_backend_free(backend);
    }
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
    Array vk_devices_idx = get_vk_devices();
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
}

void Backend::usecpu(bool p_use) {
    use_cpu = p_use;
}
bool Backend::is_use_cpu() const {
	return use_cpu;
}

void Backend::_bind_methods() {
    ClassDB::bind_method(D_METHOD("is_use_cpu"), &Backend::is_use_cpu);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_cpu"),"","is_use_cpu");
}

// SDModel
SDModel::SDModel() {
}
SDModel::~SDModel() {
    
}

void SDModel::free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != NULL) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

void SDModel::set_model_path(String p_path) {
    model_path = p_path;
}
String SDModel::get_model_path() const {
	return model_path;
}
void SDModel::set_backend(Backend p_backen) {
    backend_res = p_backen;
}
Backend SDModel::get_backend() const {
	return backend_res;
}
void SDModel::set_version(SDVersion p_version) {
    version = p_version;
}
SDVersion SDModel::get_version() const {
	return version;
}
void SDModel::set_wtype(ggml_type p_wtype){
	wtype = p_wtype
}
ggml_type SDModel::get_wtype() const {
	return wtype;
}

void SDModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_model_path"), &SDModel::get_model_path);
    ClassDB::bind_method(D_METHOD("get_backend"), &SDModel::get_backend);
    ClassDB::bind_method(D_METHOD("get_version"), &SDModel::get_version);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "", PROPERTY_USAGE_READ_ONLY),"","get_model_path");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "backend_res", PROPERTY_HINT_RESOURCE_TYPE, "Backend"), "", "get_backend");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "SDVersion", PROPERTY_HINT_ENUM, "SD1.x, SD2.x, SDXL, SVD, SD3-2B, FLUX-dev, FLUX-schnell, VERSION_COUNT", PROPERTY_USAGE_READ_ONLY),"","get_version");
    
    BIND_ENUM_CONSTANT(VERSION_SD1);
    BIND_ENUM_CONSTANT(VERSION_SD2);
	BIND_ENUM_CONSTANT(VERSION_SDXL);
    BIND_ENUM_CONSTANT(VERSION_SVD);
    BIND_ENUM_CONSTANT(VERSION_SD3_2B);
    BIND_ENUM_CONSTANT(VERSION_FLUX_DEV);
    BIND_ENUM_CONSTANT(VERSION_FLUX_SCHNELL);
    BIND_ENUM_CONSTANT(VERSION_COUNT);
}

// SDModelLoader
SDModelLoader::SDModelLoader() {
}
SDModelLoader::~SDModelLoader() {
}

Backend SDModelLoader::create_backend(int device_index, bool use_cpu = false) {
    Backend new_backend = new Backend(device_index, use_cpu);
	return new_backend;
}

void SDModelLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_backend","device_index","use_cpu"), &SDModelLoader::create_backend);
	ClassDB::bind_method(D_METHOD("load_model","backend","model_path","scheduler","vae_only_decode","vae_on_cpu","clip_on_cpu"), &SDModelLoader::load_model);

    BIND_ENUM_CONSTANT(DEFAULT);
    BIND_ENUM_CONSTANT(DISCRETE);
	BIND_ENUM_CONSTANT(KARRAS);
    BIND_ENUM_CONSTANT(EXPONENTIAL);
    BIND_ENUM_CONSTANT(AYS);
    BIND_ENUM_CONSTANT(GITS);
    BIND_ENUM_CONSTANT(N_SCHEDULES);
}


/* Model load */

Diffusion::Diffusion(String model_path,
                     Backend backend_res,
                     SDVersion version,
                     std::shared_ptr<DiffusionModel> diffusion_model,
                     std::shared_ptr<Denoiser> denoiser,
			         std::map<std::string, struct ggml_tensor*> tensors,
                     Scheduler schedule,
                     ggml_type wtype,
                     ggml_type diffusion_wtype) :
                     diffusion_wtype(diffusion_wtype),
                     diffusion_model(diffusion_model),
                     denoiser(denoiser),
                     schedule(schedule) {
    set_model_path(model_path);
    set_backend(backend_res);
    set_version(version);
    set_wtype(wtype);

    /* scale_factor */
    scale_factor     = 0.18215f;
    if (version == VERSION_SDXL) {
        scale_factor = 0.13025f;
    } else if (version == VERSION_SD3_2B) {
        scale_factor = 1.5305f;
    } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        scale_factor = 0.3611;
    } 

    /* scheduler */
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
}

Array SDModelLoader::load_model(Backend res_backend,
                                String str_model_path,
                                Scheduler schedule, 
					            
                                bool vae_only_decode,
                                bool vae_on_cpu,
                                bool clip_on_cpu) {
    /* gdscript -> C */
    ggml_type wtype = GGML_TYPE_COUNT;
    ggml_backend_t backend = res_backend.get_backend();
    const std::string model_path(str_model_path.c_str());

    /* Loader in file */
    ModelLoader model_loader;
    printlog(vformat("Loading model from '%s'", str_model_path));
    if (!model_loader.init_from_file(model_path)) {
        ERR_PRINT(vformat("Model loader failed: '%s'", str_model_path));
        return Array();
    }

    /* Get version */
    SDVersion version = model_loader.get_sd_version();
    if (version == VERSION_COUNT) {
        ERR_PRINT(vformat("Get sd version from file failed."));
        return Array();
    }
    printlog(vformat("Version: %s ", model_version_to_str[version]));
    if (version == VERSION_SDXL) {
        WARN_PRINT(vformat(
            "### It looks like you are using SDXL model. ###"
            "If you find that the generated images are completely black, "
            "try specifying SDXL VAE FP16 Fix. You can find it here: "))
        print_line_rich("[url=https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors]sdxl_vae.safetensors[/url]");
    }

    /* Get weight type */
    printlog(vformat("Get model weight type..."));
    ggml_type model_wtype           = GGML_TYPE_COUNT;
    ggml_type conditioner_wtype     = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype = GGML_TYPE_COUNT;
    ggml_type vae_wtype             = GGML_TYPE_COUNT;
    if (wtype == GGML_TYPE_COUNT) {
        model_wtype = model_loader.get_sd_wtype();
        if (model_wtype == GGML_TYPE_COUNT) {
            model_wtype = GGML_TYPE_F32;
            WARN_PRINT(vformat("Can not get model weight type from weight, use f32"));
        }
        conditioner_wtype = model_loader.get_conditioner_wtype();
        diffusion_model_wtype = model_loader.get_diffusion_model_wtype();
        vae_wtype = model_loader.get_vae_wtype();
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

    /* Creat Model */
    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;

    Backend clip_backend_res = res_backend;
    Backend vae_backend_res = res_backend;
    ggml_backend_t clip_backend.get_backend();
    ggml_backend_t vae_backend.get_backend();

    if (version == VERSION_SVD) {
        clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, conditioner_wtype);
        clip_vision->alloc_params_buffer();
        clip_vision->get_param_tensors(tensors);

        diffusion_model = std::make_shared<UNetModel>(backend, diffusion_model_wtype, version);
        diffusion_model->alloc_params_buffer();
        diffusion_model->get_param_tensors(tensors);

        first_stage_model = std::make_shared<AutoEncoderKL>(backend, vae_wtype, vae_only_decode, true, version);
        first_stage_model->alloc_params_buffer();
        first_stage_model->get_param_tensors(tensors, "first_stage_model");
    } else {
        if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
            printlog(vformat("CLIP: Using CPU"));
            clip_backend_res = create_backend(-1, true);
        }
        if (vae_on_cpu && !ggml_backend_is_cpu(backend)) {
            printlog(vformat("VAE: Using CPU"));
            vae_backend_res = create_backend(-1, true);
        }
        clip_backend = clip_backend_res.get_backend();
        vae_backend = vae_backend_res.get_backend();
        
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

        first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_wtype, vae_decode_only, false, version);
        first_stage_model->alloc_params_buffer();
        first_stage_model->get_param_tensors(tensors, "first_stage_model");
    }

    /* load weights */
    struct ggml_init_params params;
    params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10M
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
    if (vae_only_decode) {
        ignore_tensors.insert("first_stage_model.encoder");
        ignore_tensors.insert("first_stage_model.quant");
    }
    if (version == VERSION_SVD) {
        ignore_tensors.insert("conditioner.embedders.3");
    }
    bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
    if (!success) {
        ERR_PRINT(vformat("load tensors from model loader failed"));
        ggml_free(ctx);
        return Array();
    }

    /* mem_size */
    // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

    if (version == VERSION_SVD) {
        // diffusion_model->test();
        // first_stage_model->test();
        // return false;
    } else {
        size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
        size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
        size_t vae_params_mem_size  = first_stage_model->get_params_buffer_size();
        
        size_t total_params_ram_size  = 0;
        size_t total_params_vram_size = 0;
        if (ggml_backend_is_cpu(clip_backend)) {
            total_params_ram_size += clip_params_mem_size
        } else {
            total_params_vram_size += clip_params_mem_size
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

        size_t total_params_size = total_params_ram_size + total_params_vram_size;
        printlog(vformat(
            "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
            "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s)",
            total_params_size / 1024.0 / 1024.0,
            total_params_vram_size / 1024.0 / 1024.0,
            total_params_ram_size / 1024.0 / 1024.0,
            clip_params_mem_size / 1024.0 / 1024.0,
            ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
            unet_params_mem_size / 1024.0 / 1024.0,
            ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
            vae_params_mem_size / 1024.0 / 1024.0,
            ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",))
    }
    int64_t t1 = ggml_time_ms();
    printlog(vformat("loading model from '%s' completed, taking %.2fs", str_model_path, (t1 - t0) * 1.0f / 1000));

    // check is_using_v_parameterization_for_sd2
    bool is_using_v_parameterization = (version == VERSION_SD2 && is_using_v_parameterization_for_sd2(ctx));
    
    /* denoiser */
    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();
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
    } else if (version == VERSION_SVD || is_using_v_parameterization) {
        printlog(vformat("running in v-prediction mode"));
        denoiser = std::make_shared<CompVisVDenoiser>();
    } else {
        printlog(vformat("running in eps-prediction mode"));
    }

    auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
    if (comp_vis_denoiser) {
        for (int i = 0; i < TIMESTEPS; i++) {
            comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
            comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
        }
    }
    
    Array model_arr = Array();
    CLIP clip_res = new CLIP(str_model_path,
                            res_backend,
                            version,
                            conditioner_wtype,
                            cond_stage_model,
                            clip_vision);
    model_arr.push_back(clip_res);
    Diffusion diffusion_res = new Diffusion(str_model_path,
                                            res_backend,
                                            version,
                                            diffusion_model,
                                            denoiser,
                                            tensors,
                                            schedule,
                                            model_wtype,
                                            diffusion_model_wtype);
    model_arr.push_back(diffusion_res);
    VAEModel vae_res = new VAEModel(str_model_path,
                                    res_backend,
                                    version,
                                    vae_wtype,
                                    first_stage_model,
                                    nullptr,
                                    vae_only_decode);
    model_arr.push_back(vae_res);
    
    printlog(vformat("Finished loaded file"));
    ggml_free(ctx);
    return model_arr;
}
