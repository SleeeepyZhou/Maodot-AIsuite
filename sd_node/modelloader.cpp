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
