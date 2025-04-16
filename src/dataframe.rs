use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufReader, prelude::*};

#[derive(Clone, Debug)]
pub enum DataTypeValue {
    Null,
    Float(f32),
    String(String),
}

#[derive(Clone, Debug)]
pub enum DataType {
    Float,
    String,
}

#[derive(Clone)]
pub struct DataFrame {
    index_to_column: HashMap<usize, String>,
    columns: HashMap<String, (usize, DataType, Vec<DataTypeValue>)>,
    len: usize,
}

impl DataFrame {
    pub fn new() -> Self {
        return Self {
            index_to_column: HashMap::new(),
            columns: HashMap::new(),
            len: 0,
        };
    }

    pub fn from_csv(filename: &str, row_limit: Option<usize>) -> Self {
        let mut df = DataFrame::new();
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let mut reader_lines = reader.lines();
        let headers = reader_lines.next().unwrap().unwrap();
        df.insert_headers(&headers);
        let row_limit = if let Some(limit) = row_limit {
            limit
        } else {
            1000000
        };
        for (i, line) in reader_lines.enumerate() {
            if i == row_limit {
                break;
            }
            let line = line.unwrap();
            df.insert_row_as_string(&line);
        }
        return df;
    }

    fn insert_headers(&mut self, headers: &str) {
        for header in headers.split(",") {
            self.insert_header(header);
        }
    }

    fn insert_header(&mut self, header: &str) {
        let header_index = self.columns.len();
        self.columns.insert(
            header.to_string(),
            (header_index, DataType::Float, Vec::new()),
        );
        self.index_to_column
            .insert(header_index, header.to_string());
    }

    fn insert_row_as_string(&mut self, line: &str) {
        for (i, value) in line.split(",").enumerate() {
            let column_name = self.index_to_column.get(&i).unwrap().clone();
            let (_, dtype, _) = self.columns.get(&column_name).unwrap();
            if value == "" {
                self.insert_value(&column_name, DataTypeValue::Null);
            } else {
                match dtype {
                    DataType::Float => match value.parse::<f32>() {
                        Ok(parsed_value) => {
                            self.insert_value(&column_name, DataTypeValue::Float(parsed_value));
                        }
                        Err(_) => {
                            let column_name = self.index_to_column.get(&i).unwrap().clone();
                            self.convert_column_values_to_string(column_name.clone().as_str());
                            self.insert_value(
                                &column_name,
                                DataTypeValue::String(value.to_string()),
                            );
                        }
                    },
                    DataType::String => {
                        self.insert_value(&column_name, DataTypeValue::String(value.to_string()));
                    }
                }
            }
        }
        self.len += 1;
    }

    fn insert_value(&mut self, column_name: &str, value: DataTypeValue) {
        let (_, _, data) = self.columns.get_mut(column_name).unwrap();
        data.push(value)
    }

    fn convert_column_values_to_string(&mut self, column_name: &str) {
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

    pub fn data(&self) -> HashMap<&String, (&DataType, &Vec<DataTypeValue>)> {
        let mut data_hashmap = HashMap::new();
        for (column_name, (_, dtype, data)) in self.columns.iter() {
            data_hashmap.insert(column_name, (dtype, data));
        }
        return data_hashmap;
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_from_csv() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = DataFrame::from_csv(filename, Some(row_limit));
        let column_names = df.columns();
        assert!(df.len() == row_limit);
        assert!(
            column_names
                == vec![
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
        assert!(!df.columns.iter().any(|(_, (_, _, data))| {
            data.iter()
                .all(|value| matches!(value, DataTypeValue::Null))
        }));
        assert!(
            df.columns
                .iter()
                .all(|(_, (_, _, data))| { data.len() == row_limit })
        );
        assert!(df.columns.iter().all(|(_, (_, dtype, data))| {
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
            }
        }));
        assert!(df.columns.iter().all(|(column_name, (_, dtype, _))| {
            if column_name == "ocean_proximity" {
                matches!(dtype, DataType::String)
            } else {
                matches!(dtype, DataType::Float)
            }
        }));
    }

    #[test]
    fn test_get_data() {
        let filename = "housing.csv";
        let row_limit = 10;
        let df = DataFrame::from_csv(filename, Some(row_limit));
        let data = df.data();
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
        assert!(
            df.columns
                .iter()
                .all(|(_, (_, _, data))| { data.len() == row_limit })
        );
    }
}
