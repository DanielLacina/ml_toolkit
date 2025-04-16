use crate::dataframe::{DataFrame, DataType, DataTypeValue};
use std::collections::{HashMap, HashSet};

pub enum ImputerStrategy {
    Median,
}

pub enum StringEncoding {
    OneHot,
}

pub enum Scalar {
    None,
    Standard,
}

pub struct Pipeline {
    string_encoding: StringEncoding,
    imputer_strategy: ImputerStrategy,
    scalar: Scalar,
}

impl Pipeline {
    pub fn new(
        string_encoding: StringEncoding,
        imputer_strategy: ImputerStrategy,
        scalar: Scalar,
    ) -> Self {
        Self {
            string_encoding,
            imputer_strategy,
            scalar,
        }
    }

    pub fn transform(&self, df: &DataFrame) -> Vec<Vec<f32>> {
        let mut output_matrix = vec![vec![]; df.len()];
        let data = df.data();
        let mut columns_to_not_scale = HashSet::new();
        for (j, (column_name, (dtype, values))) in data.iter().enumerate() {
            match dtype {
                DataType::Float => {
                    let median = df.median(column_name);
                    for (i, value) in values.iter().enumerate() {
                        match value {
                            DataTypeValue::Float(inner) => {
                                output_matrix[i].push(*inner);
                            }
                            DataTypeValue::Null => match self.imputer_strategy {
                                ImputerStrategy::Median => {
                                    output_matrix[i].push(median);
                                }
                            },
                            _ => panic!("implementation error"),
                        }
                    }
                }
                DataType::String => match self.string_encoding {
                    StringEncoding::OneHot => {
                        let categorical_values = self.extract_categorical_values(values);
                        for (_, cat_values) in categorical_values.iter() {
                            for (i, cat_value) in cat_values.iter().enumerate() {
                                output_matrix[i].push(*cat_value);
                            }
                        }
                        columns_to_not_scale.insert(j);
                    }
                },
            }
        }
        self.scale_data(&mut output_matrix, &columns_to_not_scale);
        return output_matrix;
    }

    fn extract_categorical_values(&self, values: &Vec<DataTypeValue>) -> HashMap<String, Vec<f32>> {
        let mut categories: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            let inner = match value {
                DataTypeValue::String(inner) => inner.clone(),
                DataTypeValue::Null => "null".to_string(),
                _ => {
                    panic!("implementation error")
                }
            };
            for (category, cat_values) in categories.iter_mut() {
                if *category == *inner {
                    cat_values.push(1.0);
                } else {
                    cat_values.push(0.0);
                }
            }
            if !categories.contains_key(inner.as_str()) {
                let mut cat_values = vec![0.0; i];
                cat_values.push(1.0);
                categories.insert(inner.clone(), cat_values);
            }
        }
        return categories;
    }

    fn scale_data(&self, matrix: &mut Vec<Vec<f32>>, exclude: &HashSet<usize>) {
        match self.scalar {
            Scalar::Standard => {
                for i in (0..matrix[0].len()) {
                    if exclude.contains(&i) {
                        continue;
                    }
                    let column_vector = matrix.iter().map(|v| v[i]).collect();
                    let mean = self.mean(&column_vector);
                    let std = self.std(&column_vector, Some(mean));
                    for row_vector in matrix.iter_mut() {
                        row_vector[i] = (row_vector[i] - mean) / std;
                    }
                }
            }
            Scalar::None => {}
        }
    }

    fn mean(&self, column_vector: &Vec<f32>) -> f32 {
        let mut sum = 0.0;
        for value in column_vector {
            sum += *value;
        }
        return sum / column_vector.len() as f32;
    }

    fn std(&self, column_vector: &Vec<f32>, mean: Option<f32>) -> f32 {
        let mean = if let Some(mean) = mean {
            mean
        } else {
            self.mean(column_vector)
        };
        let mut sum = 0.0;
        for value in column_vector {
            sum += (value - mean).powf(2.0);
        }
        let std = f32::sqrt(sum / (column_vector.len() - 1) as f32);
        return std;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_transform() {
        let imputer_strategy = ImputerStrategy::Median;
        let string_encoding = StringEncoding::OneHot;
        let scalar = Scalar::Standard;
        let pipeline = Pipeline::new(string_encoding, imputer_strategy, scalar);
        let df = DataFrame::from_csv("housing.csv", Some(10000));
        let output_matrix = pipeline.transform(&df);
        assert!(output_matrix.iter().all(|v| {
             v.len() == 14
        }));
    }
}
