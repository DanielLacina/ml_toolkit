use crate::matrices;
use std::iter::zip;

pub struct LinearRegression {
    weights: Vec<f32>,
    bias: f32
} 

impl LinearRegression {

    pub fn new() -> Self {
        return Self {
            weights: Vec::new(),
            bias: 0.0
        }
    }
    pub fn fit(&mut self, data: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) {
        let mut X = data.clone(); 
        for x_vector in X.iter_mut() {
            x_vector.push(1.0);
        }
        let y = labels.clone();
        let X_transpose = matrices::transpose_matrix(&X);
        let X_output = matrices::multiply_matrices(&X_transpose, &X);
        let y_output = matrices::multiply_matrices(&X_transpose, &y);
        let X_output_inverse = matrices::inverse_matrix(&X_output);
        let parameter_matrix = matrices::multiply_matrices(&X_output_inverse, &y_output); 
        let parameters: Vec<f32> = parameter_matrix.into_iter().map(|parameter| parameter[0]).collect(); 
        let weights = parameters[0..parameters.len() - 2].to_vec();
        let bias = parameters[parameters.len() - 1];
        self.weights = weights; 
        self.bias = bias;
    }

    pub fn predict(&self, data: &Vec<Vec<f32>>) -> Vec<f32> {
        let outputs: Vec<f32> =  data.iter().map(|row_vector| zip(row_vector, self.weights.iter()).fold(0.0, |acc, (row_v, weight)| acc + row_v * weight) + self.bias ).collect();  
        return outputs;
    }
}