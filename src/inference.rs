use std::iter::zip;

pub fn mse(predictions: &Vec<f32>, labels: &Vec<f32>) -> f32 {
    let mut sum = 0.0;
    for (prediction, label) in zip(predictions, labels) {
        let sr = (label - prediction).powf(2.0);
        sum += sr;
    } 
    return sum/labels.len() as f32;
}

pub fn rmse(predictions: &Vec<f32>, labels: &Vec<f32>) -> f32 {
    f32::sqrt(mse(predictions, labels))
}


