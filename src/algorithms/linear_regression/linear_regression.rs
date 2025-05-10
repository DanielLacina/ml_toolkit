use crate::linear_algebra::{Matrix, RowVector};
use std::iter::zip;

pub struct LinearRegression {
    weights: Vec<f32>,
    bias: f32,
    ridge_value: f32,
}

impl LinearRegression {
    pub fn new(ridge_value: f32) -> Self {
        return Self {
            weights: Vec::new(),
            bias: 0.0,
            ridge_value,
        };
    }
    pub fn fit(&mut self, data: &Matrix, labels: &Matrix) {
        let X = Matrix::new(
            &data
                .matrix()
                .iter()
                .map(|vector| {
                    let mut vector = vector.vector().clone();
                    vector.push(1.0);
                    RowVector::new(&vector)
                })
                .collect(),
        );
        let y = labels.clone();
        let X_transpose = X.transpose();
        let mut X_output = X.multiply(&X_transpose);
        if self.ridge_value != 0.0 {
            X_output = Matrix::new(&X_output.matrix()
                .iter()
                .enumerate()
                .map(|(i, row_vector)| {
                    RowVector::new(&row_vector.vector()
                        .iter()
                        .enumerate()
                        .map(|(j, value)| {
                            if i == j {
                                value + self.ridge_value
                            } else {
                                *value
                            }
                        })
                        .collect::<Vec<f32>>())
                })
                .collect::<Vec<RowVector>>());
        }
        let y_output = X_transpose.multiply(&y);
        let X_output_inverse = X_output.inverse();
        let parameter_matrix = X_output_inverse.multiply(&y_output);
        let parameters: Vec<f32> = parameter_matrix
            .matrix()
            .iter()
            .map(|parameter| parameter.get(0))
            .collect();
        let weights = parameters[0..parameters.len() - 1].to_vec();
        let bias = parameters[parameters.len() - 1];
        self.weights = weights;
        self.bias = bias;
    }

    pub fn predict(&self, data: &Matrix) -> RowVector {
        let outputs: RowVector = RowVector::new(&data.
            matrix().iter()
            .map(|row_vector| {
                zip(row_vector.vector(), self.weights.iter())
                    .fold(0.0, |acc, (row_v, weight)| acc + row_v * weight)
                    + self.bias
            })
            .collect());
        return outputs;
    }

    pub fn weights(&self) -> &Vec<f32> {
        return &self.weights;
    }

    pub fn bias(&self) -> f32 {
        return self.bias;
    }
}
