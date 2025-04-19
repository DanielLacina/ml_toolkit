use crate::dataframe::datatype::{DataType, DataTypeValue};
use std::collections::{HashMap, HashSet};

const IDS: &str = "ids";

#[derive(Clone)]
pub struct DataFrame {
    index_to_column: HashMap<usize, String>,
    columns: HashMap<String, (usize, DataType, Vec<DataTypeValue>)>,
    len: usize,
}

impl DataFrame {
    pub fn new() -> Self {
        let mut df = Self {
            index_to_column: HashMap::new(),
            columns: HashMap::new(),
            len: 0,
        };
        df.insert_column(IDS, &Vec::new(), &DataType::Id);
        return df;
    }

    pub fn insert_column(
        &mut self,
        column_name: &str,
        values: &Vec<DataTypeValue>,
        dtype: &DataType,
    ) {
        assert!(values.len() == self.len());
        let header_index = self.columns.len();
        self.columns.insert(
            column_name.to_string(),
            (header_index, dtype.clone(), Vec::new()),
        );
        self.index_to_column
            .insert(header_index, column_name.to_string());
    }

    fn update_ids(&mut self) {
        let len = self.len;
        let (_, values) = self.get_column_mut(IDS);
        for i in (values.len()..len) {
            values.push(DataTypeValue::Id(i));
        }
    }

    pub fn remove_column(&mut self, column_name: &str) {
        self.columns.remove(column_name);
    }

    fn get_column_mut(&mut self, column_name: &str) -> (&mut DataType, &mut Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get_mut(column_name).unwrap();
        return (dtype, values);
    }

    pub fn get_column(&self, column_name: &str) -> (&DataType, &Vec<DataTypeValue>) {
        let (_, dtype, values) = self.columns.get(column_name).unwrap();
        return (dtype, values);
    }

    pub fn get_column_by_index(&self, index: usize) -> (&String, &DataType, &Vec<DataTypeValue>) {
        let column_name = self.index_to_column.get(&index).unwrap();
        let (dtype, values) = self.get_column(column_name);
        return (column_name, dtype, values);
    }

    pub fn insert_row(&mut self, data_hashmap: &HashMap<String, DataTypeValue>) {
        for (column_name, value) in data_hashmap.iter() {
            let (dtype, data) = self.get_column_mut(column_name);
            let right_dtype = match dtype {
                DataType::Float => {
                    matches!(value, DataTypeValue::Float(_))
                }
                DataType::Id => {
                    matches!(value, DataTypeValue::Id(_))
                }
                DataType::String => {
                    matches!(value, DataTypeValue::String(_))
                }
            };
            if !right_dtype {
                assert!(
                    matches!(value, DataTypeValue::Null),
                    "{}",
                    format!(
                        "value of {:?} is incompatible for column {} with dtype of {:?}",
                        value, column_name, dtype
                    )
                );
            }
            data.push(value.clone());
        }
        self.len += 1;
        self.update_ids();
    }

    pub fn convert_column_values_to_string(&mut self, column_name: &str) {
        let (_, dtype, data) = self.columns.get_mut(column_name).unwrap();
        for value in data.iter_mut() {
            match value {
                DataTypeValue::Float(inner) => {
                    *value = DataTypeValue::String(inner.to_string());
                }
                _ => {}
            }
        }
        *dtype = DataType::String;
    }

    pub fn columns(&self) -> Vec<&String> {
        (0..self.index_to_column.len())
            .map(|i| self.index_to_column.get(&i).unwrap())
            .collect()
    }

    pub fn get_columns_as_df(&self, columns: &Vec<String>) -> DataFrame {
        let mut df = DataFrame::new();
        for column_name in columns {
            let (dtype, data) = self.get_column(column_name);
            df.insert_column(column_name.as_str(), data, dtype);
        }
        return df;
    }

    pub fn bins(
        &self,
        column_name: &str,
        num_bins: usize,
    ) -> Vec<(DataTypeValue, DataTypeValue, u32)> {
        let (_, data) = self.get_column(column_name);
        let (_, ids) = self.get_column(IDS);
        let mut zipped: Vec<(DataTypeValue, DataTypeValue)> =
            ids.clone().into_iter().zip(data.clone()).collect();
        zipped.sort_by(|(_, a), (_, b)| a.cmp(b));
        let bin_size = zipped.len() / num_bins;
        let mut bins = Vec::new();
        for (i, (id, value)) in zipped.into_iter().enumerate() {
            bins.push((id, value, (i / bin_size) as u32));
        }
        bins.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));
        return bins;
    }

    pub fn data(&self, include_ids: bool) -> HashMap<&String, (&DataType, &Vec<DataTypeValue>)> {
        let mut data_hashmap = HashMap::new();
        for (_, column_name) in self.index_to_column.iter() {
            if column_name == IDS && !include_ids {
                continue;
            }
            let (dtype, data) = self.get_column(column_name);
            data_hashmap.insert(column_name, (dtype, data));
        }
        return data_hashmap;
    }

    pub fn median(&self, column_name: &str) -> f32 {
        let (_, values) = self.get_column(column_name);
        let mut values: Vec<f32> = values
            .iter()
            .filter_map(|value| match value {
                DataTypeValue::Float(inner) => Some(*inner),
                _ => None,
            })
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if values.len() % 2 == 0 {
            return (values[values.len() / 2] + values[(values.len() - 1) / 2]) / 2.0;
        } else {
            return values[values.len() / 2];
        }
    }

    pub fn mean(&self, column_name: &str) -> f32 {
        let mut sum = 0.0;
        let (dtype, values) = self.get_column(column_name);
        if !matches!(dtype, DataType::Float) {
            panic!("the mean can only be found from float values");
        }
        for value in values {
            match value {
                DataTypeValue::Float(inner) => {
                    sum += *inner;
                }
                _ => {}
            }
        }
        let mean = sum / values.len() as f32;
        return mean;
    }

    pub fn std(&self, column_name: &str, mean: Option<f32>) -> f32 {
        let mean = if let Some(mean) = mean {
            mean
        } else {
            self.mean(column_name)
        };
        let mut sum = 0.0;
        let (dtype, values) = self.get_column(column_name);
        if !matches!(dtype, DataType::Float) {
            panic!("the mean can only be found from float values");
        }
        for value in values {
            match value {
                DataTypeValue::Float(inner) => {
                    sum += (*inner - mean).powf(2.0);
                }
                _ => {}
            }
        }
        let std = f32::sqrt(sum / (values.len() - 1) as f32);
        return std;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv;

    #[test]
    fn test_df() {
        let filename = "housing.csv";
        let row_limit = 1000;
        let df = df_from_csv(filename, Some(row_limit));
        let column_names = df.columns();
        assert!(df.len() == row_limit);
        assert!(
            column_names
                == vec![
                    IDS,
                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",
                    "median_house_value",
                    "ocean_proximity"
                ]
        );

        assert!(
           !column_names 
                .iter().any(|column_name| {
                   let (_, data) = df.get_column(column_name);
                 data.iter()
                .all(|value| matches!(value, DataTypeValue::Null))
                }) 
        ) ;
        assert!(
           column_names 
                .iter()
                .all(|column_name| 
                {    
                    let (_, data) = df.get_column(column_name);
                    data.len() == row_limit })
        );
        assert!(column_names.iter().all(|column_name| {
            let (dtype, data) = df.get_column(column_name);
            match dtype {
                DataType::Float => data.iter().all(|value| match value {
                    DataTypeValue::Float(_) => true,
                    DataTypeValue::Null => true,
                    _ => false,
                }),
                DataType::String => data.iter().all(|value| match value {
                    DataTypeValue::String(_) => true,
                    DataTypeValue::Null => true,
                    _ => false,
                }),
                _ => true,
            }
        }));
         assert!(column_names.iter().all(|column_name| {
            let (dtype, _) = df.get_column(column_name);
            if column_name.as_str() == "ocean_proximity" {
                matches!(dtype, DataType::String)
            } else if column_name.as_str() == IDS {
                matches!(dtype, DataType::Id)   
            }  else {
                matches!(dtype, DataType::Float)
            }
        }));
    }

    #[test]
    fn test_get_data() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let data = df.data(true);
        let column_names = vec![
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "ocean_proximity",
        ];
        assert!(
            column_names
                .iter()
                .all(|column_name| data.contains_key(&column_name.to_string()))
        );
        assert!(column_names.iter().all(|column_name| {
            let (_, data) = df.get_column(column_name);
            data.len() == row_limit
        }));
    }

    #[test]
    fn test_get_mean() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let mut means: HashMap<String, f32> = HashMap::new();
        means.insert("longitude".to_string(), -122.24500);
        means.insert("latitude".to_string(), 37.85000);
        means.insert("housing_median_age".to_string(), 46.80000);
        means.insert("total_rooms".to_string(), 2500.90000);
        means.insert("total_bedrooms".to_string(), 470.10000);
        means.insert("population".to_string(), 976.30000);
        means.insert("households".to_string(), 458.20000);
        means.insert("median_income".to_string(), 4.99608);
        means.insert("median_house_value".to_string(), 314480.00000);
        assert!(
            means
                .iter()
                .all(|(column_name, mean)| (df.mean(column_name) - mean).abs() < 0.01)
        );
    }

    #[test]
    fn test_get_std() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));

        let mut stds: HashMap<String, f32> = HashMap::new();
        stds.insert("longitude".to_string(), 0.011785);
        stds.insert("latitude".to_string(), 0.012472);
        stds.insert("housing_median_age".to_string(), 10.064238);
        stds.insert("total_rooms".to_string(), 1858.210456);
        stds.insert("total_bedrooms".to_string(), 315.910483);
        stds.insert("population".to_string(), 648.036702);
        stds.insert("households".to_string(), 323.469662);
        stds.insert("median_income".to_string(), 2.243359);
        stds.insert("median_house_value".to_string(), 68355.310287);
        assert!(stds.iter().all(|(column_name, std)| {
            let df_std = df.std(column_name, None);
            (df_std - std).abs() < 0.01
        }));
    }

    #[test]
    fn test_get_median() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = df_from_csv(filename, Some(row_limit));
        let mut medians: HashMap<String, f32> = HashMap::new();
        medians.insert("longitude".to_string(), -122.2500);
        medians.insert("latitude".to_string(), 37.8500);
        medians.insert("housing_median_age".to_string(), 52.0000);
        medians.insert("total_rooms".to_string(), 2081.0000);
        medians.insert("total_bedrooms".to_string(), 384.5000);
        medians.insert("population".to_string(), 829.5000);
        medians.insert("households".to_string(), 386.5000);
        medians.insert("median_income".to_string(), 3.9415);
        medians.insert("median_house_value".to_string(), 320250.0000);
        assert!(medians.iter().all(|(column_name, median)| {
            let df_median = df.median(column_name);
            (df_median - median).abs() < 0.01
        }));
    }

    #[test]
    fn test_get_bins() {
        let filename = "housing.csv";
        let row_limit = 10;
        let bins = 5;
        let bin_size = row_limit/bins; 
        let df = df_from_csv(filename, Some(row_limit));
        let mut bins = df.bins("population", bins);
        bins.sort_by(|(_, _, a), (_, _, b)| a.cmp(b)); 
        assert!(bins.iter().enumerate().all(|(i, (_, a_value, bin_value))| {
             let bin_row_count_by_bin = i/(*bin_value as usize + 1); 
             let bin_cmp_start = (i as i32 - bin_row_count_by_bin as i32 - bin_size as i32).max(0) as usize;
             let bin_cmp_end = i - bin_row_count_by_bin; 
             for (_, b_value, _) in bins[bin_cmp_start..bin_cmp_end].iter() {
                if !(a_value >= b_value) {
                    return false;
                }
             }  
             true 
        }));
    }
}
