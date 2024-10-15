/*       By SleeeepyZhou        */

#ifndef COND_H
#define COND_H

#include "stablediffusion.h"

class SDCond : public SDResource {
	GDCLASS(SDCond, SDResource);

private:
    int clip_skip = -1;

    int width = 512;
	int height = 512;
    Ref<Image> input_image;
    LatentFromImage create_mode = BASE_IMAGE;
    ggml_tensor* latent;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	void set_width(const int &p_width);
	int get_width() const;
    void set_height(const int &p_height);
	int get_height() const;
    void set_image(const Ref<Image> &p_image);
	Ref<Image> get_image() const;
    void set_create_mode(LatentFromImage p_mode);
	LatentFromImage get_create_mode() const;

    void creat_latent();
    void free_latent();

};

class Control : public StableDiffusion {
	GDCLASS(Control, StableDiffusion);


    SDCond text_encoders();
}

#endif // COND_H