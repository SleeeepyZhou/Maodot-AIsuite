/*  StableDiffusion for Modot   */
/*       By SleeeepyZhou        */

/* System */
#include "deviceinfo.h"

/* GGML */
#include "ggml_extend.hpp"

/* StableDiffusion */
#include "conditioner.hpp"

/* Module header */
#include "stablediffusion.h"

#include "sdmodel.h"
#include "sdcond.h"
#include "ksampler.h"
#include "latent.h"


//// Base class : stablediffusion

// SDResource
SDResource::SDResource() {
}
SDResource::~SDResource() {
}

void SDResource::_bind_methods() {
}

// StableDiffusion
StableDiffusion::StableDiffusion() {
}
StableDiffusion::~StableDiffusion() {
}

void StableDiffusion::set_n_threads(bool p_threads) {
    n_threads = p_threads;
}
int StableDiffusion::get_n_threads() const {
	return n_threads;
}

Array StableDiffusion::get_vk_devices() const {
	return DeviceInfo::getInstance().get_available_devices();
}

void StableDiffusion::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_n_threads", "threads"), &StableDiffusion::set_n_threads);
    ClassDB::bind_method(D_METHOD("get_n_threads"), &StableDiffusion::get_n_threads);

    ClassDB::bind_method(D_METHOD("get_vk_devices"), &StableDiffusion::get_vk_devices);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads", PROPERTY_HINT_RANGE, "-1,512"), "set_n_threads", "get_n_threads");
}


//// sdcond

/*

Res
SDcond

Node
Control
    in  CLIP
        prompt
        cn
        cimage

    out SDcond

*/

// SDCond
SDCond::SDCond(SDCondition cond, SDCondition uncond):
                cond(cond), uncond(uncond) {
}
SDCond::~SDCond() {
}

SDCondition SDCond::get_cond() const {
	return cond;
}
SDCondition SDCond::get_uncond() const {
	return uncond;
}

void SDCond::_bind_methods() {
}


// SDControl
SDControl::SDControl() {
}
SDControl::~SDControl() {
}

SDCond SDControl::get_cond_res() const {
	return sdcond;
}

void SDControl::text_encoders(SDModel model_node, Latent latent , 
                              String prompt, String negative_prompt, 
                              int clip_skip) {
    // Get condition
    Ref<StableDiffusionGGML> sd = model_node.sd;
    if (!sd.is_valid()) {
        ERR_PRINT(vformat("No model is loaded."));
        return;
    }
    Array latent_info = latent.get_latent_info();
    int width;
    int height;
    if (!latent_info[0]) { 
        ERR_PRINT(vformat("No latent."));
        return;
    } else {
        width = latent_info[1];
        height = latent_info[2];
    }
    struct ggml_context* work_ctx = latent.get_work_ctx();
    int64_t t0 = ggml_time_ms();
    SDCondition cond = sd->cond_stage_model->get_learned_condition(work_ctx,
                                                            n_threads,
                                                            prompt,
                                                            clip_skip,
                                                            width,
                                                            height,
                                                            sd->diffusion_model->get_adm_in_channels());
    bool force_zero_embeddings = false;
    if (sd->version == VERSION_SDXL && negative_prompt.is_empty()) {
        force_zero_embeddings = true;
    }
    SDCondition uncond = sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                n_threads,
                                                                negative_prompt,
                                                                clip_skip,
                                                                width,
                                                                height,
                                                                sd->diffusion_model->get_adm_in_channels(),
                                                                force_zero_embeddings);
    int64_t t1 = ggml_time_ms();
    printlog(vformat("Get learned condition completed, taking %" PRId64 " ms", t1 - t0));
    SDCond condition = new SDCond(cond,uncond);
    sdcond = condition;
	emit_signal(SNAME("encode_log"), condition);
}

void SDControl::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_cond_res"), &SDControl::get_cond_res);
    ClassDB::bind_method(D_METHOD("text_encoders","model_node","latent","prompt","negative_prompt","clip_skip"), &SDControl::text_encoders);

    ADD_SIGNAL(MethodInfo("encode_log", PropertyInfo(Variant::OBJECT, "sdcond_res")));
}


//// Latent

Latent::Latent() {
}
Latent::~Latent() {
    free_work_ctx()
}

void Latent::set_width(const int &p_width) {
    width = p_width
}
int Latent::get_width() const {
	return width;
}
void Latent::set_height(const int &p_height) {
    height = p_height
}
int Latent::get_height() const {
	return height;
}
void Latent::set_batch_count(const int &p_count) {
    batch_count = p_count;
}
int Latent::get_batch_count() const {
	return batch_count;
}
Array Latent::get_latent_info() const {
    Array info;
    if (latent == NULL) {
        info.push_back(false);
        info.push_back(0);
        info.push_back(0);
        info.push_back(-1);
        info.push_back("");
    } else {
        info.push_back(true);
        info.push_back(latent_width);
        info.push_back(latent_height);
        info.push_back(latent_batch_count);
        info.push_back(vformat("%.2fMB", work_mem));
    }
    return info;
}

struct ggml_context *Latent::get_work_ctx() const {
	return work_ctx;
}
struct ggml_tensor *Latent::get_latent() const {
	return latent;
}
void Latent::take_result_latent(std::vector<struct ggml_tensor*> result_latents) {
    has_result = true;
    final_latents = result_latents;
    emit_signal(SNAME("result_latent"));
}
std::vector<struct ggml_tensor *> Latent::get_final_latents() const {
	return final_latents;
}
void Latent::free_work_ctx() {
    if (work_ctx) {
        ggml_free(work_ctx);
        work_ctx = NULL;
        latent = NULL;
    }
}

void Latent::create_latent(SDVersion version) {
    Array result;
    if (latent) {free_work_ctx()}

    /* Context create */
    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (version == VERSION_SD3_2B) {
        params.mem_size *= 3;
    }
    if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        params.mem_size *= 4;
    }
    params.mem_size += width * height * 3 * sizeof(float);
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);
    work_ctx = ggml_init(params);
    if (!work_ctx) {
        ERR_PRINT(vformat("Context create failed"));
        result.push_back(false);
        result.push_back(vformat("Context create failed"));
        emit_signal(SNAME("latent_log"), result);
        return;
    }
    latent_height = height;
    latent_width = width;
    latent_batch_count = batch_count;
    work_mem = params.mem_size / 1024.0 / 1024.0;

    /* Latent create */
    int C = 4;
    if (version == VERSION_SD3_2B) {
        C = 16;
    } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;
    latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    if (version == VERSION_SD3_2B) {
        ggml_set_f32(latent, 0.0609f);
    } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        ggml_set_f32(latent, 0.1159f);
    } else {
        ggml_set_f32(latent, 0.f);
    }
    printlog(vformat("Latent created"));
    result.push_back(true);
    result.push_back(vformat("Latent created. Work context memory size = %.2fMB", work_mem));
    has_result = false;
    emit_signal(SNAME("latent_log"), result);
}

void Latent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_width", "width"), &Latent::set_width);
    ClassDB::bind_method(D_METHOD("get_width"), &Latent::get_width);
    ClassDB::bind_method(D_METHOD("set_height", "height"), &Latent::set_height);
    ClassDB::bind_method(D_METHOD("get_height"), &Latent::get_height);
    ClassDB::bind_method(D_METHOD("set_batch_count", "batch_count"), &Latent::set_batch_count);
    ClassDB::bind_method(D_METHOD("get_batch_count"), &Latent::get_batch_count);
    ClassDB::bind_method(D_METHOD("get_latent_info"), &Latent::get_latent_info);

    ClassDB::bind_method(D_METHOD("create_latent", "sd_version"), &Latent::create_latent);

    ADD_SIGNAL(MethodInfo("result_latent"));
    ADD_SIGNAL(MethodInfo("latent_log", PropertyInfo(Variant::ARRAY, "latent_info")));

    ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "32,4096"),"set_width","get_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "32,4096"),"set_height","get_height");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "batch_count", PROPERTY_HINT_RANGE, "1,50"),"set_batch_count","get_batch_count");

}


/*
//// vae



Node
VAE
        bool    decode
    in  VAE     AutoEncoderKL / TinyAutoEncoder
        latent  /   image
    out image   /   latent



// VAE
VAE::VAE() {
}
VAE::~VAE() {

}

void VAE::set_in_image(const Ref<Image> &p_image) {
    input_image = p_image
}
Ref<Image> VAE::get_in_image() const {
	return input_image;
}

// ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
ggml_tensor* VAE::get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
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
ggml_tensor* VAE::compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
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

ggml_tensor* VAE::encode_first_stage(Latent latent) {
    struct ggml_context* work_ctx = latent.get_work_ctx();
    ggml_tensor* x = latent.get_latent();
    return compute_first_stage(work_ctx, x, false);
}
ggml_tensor* VAE::decode_first_stage(Latent latent) {
    struct ggml_context* work_ctx = latent.get_work_ctx();
    ggml_tensor* x = latent.get_latent();
    return compute_first_stage(work_ctx, x, true);
}

void VAE::_bind_methods() {

}
*/