use image::{GenericImageView};
use std::path::Path;
use tch::Tensor;
use tch::{CModule, Kind};

fn load_and_preprocess_image(image_path: &str, image_size: i64) -> Tensor {
    let img = image::open(&Path::new(image_path)).expect("Failed to open image");
    let img = img.resize_exact(image_size as u32, image_size as u32, image::imageops::FilterType::Nearest);

    let img_rgb = img.to_rgb8();
    let img_vec: Vec<f32> = img_rgb.pixels().flat_map(|p| p.0).map(|p| p as f32 / 255.0).collect();

    Tensor::f_from_slice(&img_vec).unwrap().view([1, 3, image_size, image_size])
}

fn load_model(model_path: &str) -> CModule {
    CModule::load(model_path).expect("Failed to load model")
}

fn classify_image(model: &CModule, image_tensor: Tensor) -> i64 {
    let output = model.forward_ts(&[image_tensor]).unwrap();
    let probabilities = output.softmax(-1, Kind::Float);
    let (predicted_class, _) = probabilities.max_dim(1, false);

    predicted_class.int64_value(&[0])
}

fn main() {
    let model_path = "path/to/your/trained_model.pt";
    let image_path = "path/to/plant_leaf_image.jpg";

    let model = load_model(model_path);
    let image_tensor = load_and_preprocess_image(image_path, 224); // Assuming 224x224 image size
    let class = classify_image(&model, image_tensor);

    println!("Predicted class: {}", class);
}
