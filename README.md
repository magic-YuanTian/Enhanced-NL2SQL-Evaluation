# Enhanced NL2SQL Evaluation Framework

A robust framework for evaluating Natural Language to SQL (NL2SQL) models using synthetic data and execution accuracy metrics. This framework goes beyond simple string matching by creating populated databases and comparing actual query execution results.

## Features

- **Synthetic Data Generation**: Automatically creates realistic test data based on schema structure
- **Smart Value Extraction**: Analyzes SQL queries to identify and preserve important test values
- **Foreign Key Handling**: Maintains referential integrity in generated data
- **Enhanced Execution Accuracy**: Provides detailed comparison metrics between gold and predicted SQL queries
- **Flexible Schema Support**: Works with complex database schemas including foreign key relationships
- **Sandbox Environment**: Creates temporary SQLite databases for safe query execution

## Installation

```bash
pip install sqlite3 uuid pandas
```

## Usage

The framework requires three main inputs:
1. Database schema
2. Gold (correct) SQL query
3. Predicted SQL query

### Basic Example

```python
from nl2sql_eval import enhanced_execution_accuracy

# Your database schema
schema = {
    "users": {
        "columns": [
            {"field": "id", "type": "text"},
            {"field": "name", "type": "text"},
            {"field": "age", "type": "integer"}
        ]
    }
}

# SQL queries to compare
sql_gold = "SELECT name FROM users WHERE age > 25"
sql_predicted = "SELECT name FROM users WHERE age >= 25"

# Get accuracy metrics
result = enhanced_execution_accuracy(sql_gold, sql_predicted, schema)
```

### Schema Format

The schema should be provided in the following format:

```json
{
    "table_name": {
        "columns": [
            {
                "field": "column_name",
                "type": "data_type",
                "foreign_ref": "referenced_table(column)" // Optional
            }
        ]
    }
}
```

## Output Metrics

The framework provides comprehensive comparison metrics:

- **Precision**: Accuracy of predicted results compared to gold standard
- **Recall**: Coverage of gold standard results in prediction
- **F1 Score**: Harmonic mean of precision and recall
- **Result Exact Match**: Boolean indicating exact match between results
- **Column Match**: Whether the result sets have identical columns
- **Data Match**: Whether the actual data matches exactly
- **Column Coverage**: Percentage of required columns present in prediction
- **Result Counts**: Numbers of gold, predicted, and correct results

Example output:
```python
{
    "precision": 0.95,
    "recall": 0.92,
    "f1_score": 0.93,
    "result_exact_match": False,
    "column_match": True,
    "data_match": False,
    "column_coverage": 1.0,
    "gold_count": 100,
    "predicted_count": 98,
    "correct_count": 95,
    "gold_columns": {"name", "age"},
    "pred_columns": {"name", "age"},
    "common_columns": {"name", "age"},
    "all_columns": {"name", "age"}
}
```

## Key Components

1. **Data Generation**: Creates synthetic data while preserving specified values from queries
2. **Query Analysis**: Extracts important values and conditions from SQL queries
3. **Database Management**: Handles temporary database creation and cleanup
4. **Result Comparison**: Provides detailed metrics on query execution results

## Error Handling

The framework includes robust error handling for:
- Invalid SQL queries
- Schema validation
- Database operations
- Data type mismatches
- Foreign key constraints

## Limitations

- Currently supports SQLite databases only
- Requires valid SQL syntax for both queries
- Performance may vary with very large schemas
- Limited support for complex SQL features (e.g., window functions)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
