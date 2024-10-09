// SD Code
#include "stable-diffusion.h"

// Mod
#include "stablediffusion.h"

// Engine
#include "core/io/image.h"

StableDiffusion::StableDiffusion() {
}
StableDiffusion::~StableDiffusion() {
}

/*
sd_ctx_t* new_sd_ctx(const char* model_path,
                            const char* clip_l_path,
                            const char* t5xxl_path,
                            const char* diffusion_model_path,
                            const char* vae_path,
                            const char* taesd_path,
                            const char* control_net_path_c_str,
                            const char* lora_model_dir,
                            const char* embed_dir_c_str,
                            const char* stacked_id_embed_dir_c_str,
                            bool vae_decode_only,
                            bool vae_tiling,
                            bool free_params_immediately,
                            int n_threads,
                            enum sd_type_t wtype,
                            enum rng_type_t rng_type,
                            enum schedule_t s,
                            bool keep_clip_on_cpu,
                            bool keep_control_net_cpu,
                            bool keep_vae_on_cpu);
*/
Error StableDiffusion::load_model(String model_path) {
    const char* p_model_path = model_path.utf8().get_data()
    
    sd_ctx_t* sd_ctx = new_sd_ctx(p_model_path,
                                  params.clip_l_path.c_str(),
                                  params.t5xxl_path.c_str(),
                                  params.diffusion_model_path.c_str(),
                                  params.vae_path.c_str(),
                                  params.taesd_path.c_str(),
                                  params.controlnet_path.c_str(),
                                  params.lora_model_dir.c_str(),
                                  params.embeddings_path.c_str(),
                                  params.stacked_id_embeddings_path.c_str(),
                                  true,
                                  params.vae_tiling,
                                  true,
                                  params.n_threads,
                                  params.wtype,
                                  params.rng_type,
                                  params.schedule,
                                  params.clip_on_cpu,
                                  params.control_net_cpu,
                                  params.vae_on_cpu);
	
	if (sd_ctx == NULL) {
        ERR_PRINT("new_sd_ctx_t failed\n");
        return NULL;
    }
}

Ref<Image> StableDiffusion::t2i(String model_path, String prompt){

}

void StableDiffusion::_bind_methods() {
	ClassDB::bind_method(D_METHOD("t2i", "model_path", "prompt"), &StableDiffusion::t2i);
    // ClassDB::bind_method(D_METHOD("request", "url", "custom_headers", "method", "request_data"), 
    //      &HTTPRequest::request, DEFVAL(PackedStringArray()), DEFVAL(HTTPClient::METHOD_GET), DEFVAL(String()));
    
    // ADD_SIGNAL(MethodInfo("sampling_done", PropertyInfo(Variant::Ref<Image>, "result")));
}
