/* StableDiffusion */


/* Module header */
#include "stablediffusion.h"

#include "modelloader.h"
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

void StableDiffusion::printlog(String p_log) {
    if print_log {
        print_line(log);
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

Latent KSampler::sampling(

                            ) {
    
    return                             
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

void Latent::set_image(const Ref<Image> &p_image) {
    input_image = p_image
}

Ref<Image> Latent::get_image() const {
	return input_image;
}

void Latent::set_create_mode(LatentFromImage p_mode) {
    create_mode = p_mode
}

LatentFromImage Latent::get_create_mode() const {
	return create_mode;
}

void Latent::creat_latent() {
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

VAE::VAE() {

}

VAE::~VAE() {

}

void VAE::_bind_methods() {

}

/*
// ClassDB::bind_method(D_METHOD("t2i", "model_path", "prompt"), &StableDiffusion::t2i);
// ClassDB::bind_method(D_METHOD("request", "url", "custom_headers", "method", "request_data"), 
//      &HTTPRequest::request, DEFVAL(PackedStringArray()), DEFVAL(HTTPClient::METHOD_GET), DEFVAL(String()));
// ADD_SIGNAL(MethodInfo("sampling_done", PropertyInfo(Variant::Ref<Image>, "result")));

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
Error StableDiffusion::load_model(String model_path) {
    const char* p_model_path = model_path.utf8().get_data();
    
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
*/


