use super::encoders::encoder::Encoder;
use super::imputers::imputer::Imputer;
use super::scalars::scalar::Scalar;
use crate::dataframe::{DataFrame, DataType, DataTypeValue};

pub trait Pipeline {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame;
}

pub struct NumericalPipeline {
    imputer: Imputer,
    scalar: Option<Box<dyn Scalar>>,
}

impl NumericalPipeline {
    pub fn new(imputer: Imputer, scalar: Option<Box<dyn Scalar>>) -> Self {
        return Self { imputer, scalar };
    }
}

impl Pipeline for NumericalPipeline {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let df_filled = self.imputer.fill(df, column_names);
        if let Some(scalar) = self.scalar.as_ref() {
            let df_scaled = scalar.scale_data(&df_filled, column_names);
            return df_scaled;
        } else {
            return df_filled;
        }
    }
}

pub struct CategoricalPipeline {
    encoder: Box<dyn Encoder>,
}

impl CategoricalPipeline {
    pub fn new(encoder: Box<dyn Encoder>) -> Self {
        return Self { encoder };
    }
}

impl Pipeline for CategoricalPipeline {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df_encoded = self.encoder.apply_encoding(df, column_names);
        for column_name in column_names {
            df_encoded.remove_column(column_name);
        }
        return df_encoded;
    }
}

pub struct ColumnTransformer {
    num_pipeline: NumericalPipeline,
    categorical_pipeline: CategoricalPipeline,
}

impl ColumnTransformer {
    pub fn new(num_pipeline: NumericalPipeline, categorical_pipeline: CategoricalPipeline) -> Self {
        Self {
            num_pipeline,
            categorical_pipeline,
        }
    }
    pub fn transform(&self, df: &DataFrame) -> Vec<Vec<f32>> {
        let categorical_columns = df
            .categorical_columns()
            .into_iter()
            .map(|column_name| column_name.clone())
            .collect();
        let df_transformed = self
            .categorical_pipeline
            .transform(df, &categorical_columns);
        let numeric_columns = df
            .numeric_columns()
            .into_iter()
            .map(|column_name| column_name.clone())
            .collect();
        let df_transformed = self
            .num_pipeline
            .transform(&df_transformed, &numeric_columns);
        let output_matrix = df_transformed.as_matrix(false);
        return output_matrix;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataframe::csv::df_from_csv,
        pipeline::{
            encoders::one_hot_encoder::{self, OneHotEncoder},
            imputers::imputer::{Imputer, ImputerStrategy},
            scalars::standard_scalar::StandardScalar,
        },
    };

    #[test]
    fn test_pipeline_transform() {
        let encoder = OneHotEncoder::new();
        let scalar = StandardScalar::new();
        let imputer = Imputer::new(&ImputerStrategy::Median);
        let categorical_pipeline = CategoricalPipeline::new(Box::new(encoder));
        let numeric_pipeline = NumericalPipeline::new(imputer, Some(Box::new(scalar)));
        let pipeline = ColumnTransformer::new(numeric_pipeline, categorical_pipeline);
        let df = df_from_csv("housing.csv", Some(10000));
        let output_matrix = pipeline.transform(&df);
        assert!(output_matrix.iter().all(|v| { v.len() == 14 }));
    }
}
