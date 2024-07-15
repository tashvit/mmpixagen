from PIL import Image


def turn_magenta(input_img_path, output_img_path):
    # Convert input image to RGBA
    img = Image.open(input_img_path).convert("RGBA")

    # New image with a magenta background
    magenta_bg = Image.new("RGBA", img.size, (255, 0, 255, 255))

    # Get input image on top of magenta image
    img_with_magenta_bg = Image.alpha_composite(magenta_bg, img)

    # Convert back to RGB to remove alpha channel
    final_img = img_with_magenta_bg.convert("RGB")

    # Save image to output directory
    final_img.save(output_img_path)
