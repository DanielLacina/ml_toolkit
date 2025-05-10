use super::transformers::Transformer;
use crate::dataframe::{DataFrame, DataType, DataTypeValue};
use crate::linear_algebra::Matrix;

pub struct NumericalPipeline {
    transformers: Vec<Box<dyn Transformer>>,
}

impl NumericalPipeline {
    pub fn new(transformers: Vec<Box<dyn Transformer>>) -> Self {
        return Self { transformers };
    }
}

impl Transformer for NumericalPipeline {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        for transformer in self.transformers.iter() {
            df = transformer.transform(&df, column_names);
        }
        return df;
    }
}

pub struct CategoricalPipeline {
    transformers: Vec<Box<dyn Transformer>>,
}

impl CategoricalPipeline {
    pub fn new(transformers: Vec<Box<dyn Transformer>>) -> Self {
        return Self { transformers };
    }
}

impl Transformer for CategoricalPipeline {
    fn transform(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame {
        let mut df = df.clone();
        for transformer in self.transformers.iter() {
            df = transformer.transform(&df, column_names);
        }
        return df;
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
    pub fn transform(&self, df: &DataFrame) -> Matrix {
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
            transformers::Transformer,
        },
    };

    #[test]
    fn test_pipeline_transform() {
        let encoder: Box<dyn Transformer> = Box::new(OneHotEncoder::new(false));
        let scalar: Box<dyn Transformer> = Box::new(StandardScalar::new());
        let imputer: Box<dyn Transformer> = Box::new(Imputer::new(&ImputerStrategy::Median));
        let categorical_transformers = vec![encoder];
        let categorical_pipeline = CategoricalPipeline::new(categorical_transformers);
        let numeric_transformers = vec![imputer, scalar];
        let numeric_pipeline = NumericalPipeline::new(numeric_transformers);
        let pipeline = ColumnTransformer::new(numeric_pipeline, categorical_pipeline);
        let df = df_from_csv("housing.csv", Some(10000));
        let output_matrix = pipeline.transform(&df);
        assert!(output_matrix.matrix().iter().all(|v| { v.len() == 14 }));
    }
}
