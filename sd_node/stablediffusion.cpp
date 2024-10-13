/* StableDiffusion */
#include "ggml_extend.hpp"

/* Module header */
#include "stablediffusion.h"

#include "modelloader.h"
#include "sdcond.h"
#include "vae_node.h"
#include "ksampler.h"
#include "latent.h"

//// Base class : stablediffusion

SDResource::SDResource() {
}
SDResource::~SDResource() {
}

StableDiffusion::StableDiffusion() {
}
StableDiffusion::~StableDiffusion() {
}

void StableDiffusion::printlog(String out_log) {
    if (print_log) {
        print_line(out_log);
    }
}

void StableDiffusion::set_print_log(bool p_print_log) {
	print_log = p_print_log;
}

bool StableDiffusion::is_print_log() const {
	return print_log;
}

void StableDiffusion::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_print_log", "enable"), &StableDiffusion::set_print_log);
    ClassDB::bind_method(D_METHOD("is_print_log"), &StableDiffusion::is_print_log);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "print_log"), "set_print_log", "is_print_log");
}


//// modelloader

/*

Res
SDModel

Node
SDModelLoader
    in  lora
        model_path
    
    out SDModel
        VAE
        _CLIP

*/

SDModel::SDModel() {
}

SDModel::~SDModel() {

}

void SDModel::set_model(const String &p_model_path) {

}

void SDModel::_set_model_path(const String &p_model_path) {

}

String SDModel::_get_model_path() const {
	return String();
}

void SDModel::_bind_methods() {
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_model_path", "_get_model_path");

}


SDModelLoader::SDModelLoader() {
    SDModel = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
}

SDModelLoader::~SDModelLoader() {
    free_sd_ctx(SDModel);
}

void SDModelLoader::set_schedule(Scheduler p_schedule) {
	schedule = p_schedule;
}

SDModelLoader::Scheduler SDModelLoader::get_schedule() const {
	return schedule;
}

void SDModelLoader::load_model(String model_path) {
    
}

void SDModelLoader::free_model() {
    free_sd_ctx(SDModel);
    SDModel = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if print_log {
        print_line("Model freed")
    } 
}

void SDModelLoader::_bind_methods() {
    ClassDB::bind_method(D_METHOD("free_model"), &SDModelLoader::free_model);
    ClassDB::bind_method(D_METHOD("load_model", "model_path"), &SDModelLoader::load_model);
    ClassDB::bind_method(D_METHOD("set_schedule", "scheduler"), &SDModelLoader::set_schedule);
    ClassDB::bind_method(D_METHOD("get_schedule"), &SDModelLoader::get_schedule);
    
	ADD_PROPERTY(PropertyInfo(Variant::INT, "schedule", PROPERTY_HINT_ENUM, "DEFAULT,DISCRETE,KARRAS,EXPONENTIAL,AYS,GITS,N_SCHEDULES"), "set_schedule", "get_schedule");
}


//// ksampler

/*

Node
KSampler
        latent
    in  context
        sdmodel

    out latent

*/

KSampler::KSampler() {
}

KSampler::~KSampler() {
}

void KSampler::set_modelloader(const NodePath &p_node_a) {
	if (a == p_node_a) {
		return;
	}
/*
	if (is_configured()) {
		_disconnect_signals();
	}

	a = p_node_a;
	if (Engine::get_singleton()->is_editor_hint()) {
		// When in editor, the setter may be called as a result of node rename.
		// It happens before the node actually changes its name, which triggers false warning.
		callable_mp(this, &Joint2D::_update_joint).call_deferred(false);
	} else {
		_update_joint();
	}
*/
}

NodePath KSampler::get_modelloader() const {
	return modelloader;
}

void KSampler::sample(Latent init_latent) {
}


void KSampler::_bind_methods() {
}


//// latent

/*

Res
latent

*/

Latent::Latent() {
}
Latent::~Latent() {
    ggml_free(work_ctx);
    printlog(vformat("Work context free."))
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

void Latent::create_latent(SDVersion version) {
    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (version == VERSION_SD3_2B) {
        params.mem_size *= 3;
    }
    if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
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

    work_ctx = ggml_init(params);
    if (!work_ctx) {
        ERR_PRINT("Context create failed");
        return NULL;
    }

    int C = 4;
    if (version == VERSION_SD3_2B) {
        C = 16;
    } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;
    ggml_tensor* init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    if (version == VERSION_SD3_2B) {
        ggml_set_f32(init_latent, 0.0609f);
    } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL) {
        ggml_set_f32(init_latent, 0.1159f);
    } else {
        ggml_set_f32(init_latent, 0.f);
    }
}

ggml_tensor *Latent::get_latent() const {
	return latent;
}

ggml_context *Latent::get_work_ctx() const {
	return work_ctx;
}

void Latent::free_latent() {
}

void Latent::_bind_methods() {

}

//// vae

/*

Node
VAE
        bool    decode
    in  VAE     AutoEncoderKL / TinyAutoEncoder
        latent  /   image
    out image   /   latent

*/

VAEModel::VAEModel() {
}

VAEModel::~VAEModel() {
}

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

