extends StableDiffusion

var image : Image:
	set(i):
		$TextureRect.texture = ImageTexture.create_from_image(i)

func _on_button_pressed() -> void:
	print(t2i("E:/Godot/Engine/Maodot-AIsuite/.test/sd-vulkan-x64/allInOnePixelModel_v1.ckpt",
				"a cat"))
