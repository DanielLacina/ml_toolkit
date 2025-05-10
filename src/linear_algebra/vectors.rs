use std::iter::zip;

#[derive(Clone, PartialEq, Debug)]
pub struct RowVector {
    vector: Vec<f32>,
}

impl RowVector {
    pub fn new(vector: &Vec<f32>) -> Self {
        Self {
            vector: vector.clone(),
        }
    }

    pub fn vector(&self) -> &Vec<f32> {
        &self.vector
    }

    pub fn norm(&self) -> f32 {
        self.vector.iter().fold(0.0, |acc, c| acc + c)
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn multiply_by_scalar(&self, scalar: f32) -> RowVector {
        let scaled_vector = self.vector().iter().map(|c| c * scalar).collect();
        RowVector::new(&scaled_vector)
    }

    pub fn add_vector(&self, v2: &RowVector) -> RowVector {
        let summed_vectors: Vec<f32> = zip(self.vector(), v2.vector())
            .map(|(c1, c2)| *c1 + *c2)
            .collect();
        RowVector::new(&summed_vectors)
    }

    pub fn get(&self, i: usize) -> f32 {
        self.vector[i].clone()
    }
}
