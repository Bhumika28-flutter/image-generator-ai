from flask import Flask, render_template, request
import generate  # your generate.py file
import os

app = Flask(__name__)
pipe = generate.load_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form["prompt"]
    seed = request.form.get("seed", None)
    width = int(request.form.get("width", 512))
    height = int(request.form.get("height", 512))
    steps = int(request.form.get("steps", 20))
    scale = float(request.form.get("scale", 7.5))
    num_images = int(request.form.get("num_images", 1))

    seed = int(seed) if seed and seed.isdigit() else None

    # Generate the image(s)
    success = generate.generate_images(
        pipe,
        prompt,
        num_images=num_images,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        scale=scale
    )

    if success:
        image_files = sorted(os.listdir("images"), key=lambda x: os.path.getmtime(os.path.join("images", x)), reverse=True)[:num_images]
        return render_template("index.html", images=image_files)
    else:
        return "❌ Image generation failed!", 500

if __name__ == "__main__":
    app.run(debug=True)
