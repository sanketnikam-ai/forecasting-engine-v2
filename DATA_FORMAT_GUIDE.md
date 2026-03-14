# Data Format Guide - YYYYMM Format

## What is YYYYMM Format?

YYYYMM is a compact date format where:
- **YYYY** = 4-digit year
- **MM** = 2-digit month (01-12)

### Examples:
```
202301 = January 2023
202302 = February 2023
202312 = December 2023
202401 = January 2024
```

## Why YYYYMM Format?

✅ **Advantages:**
- Simple and compact
- Easy to sort chronologically
- No ambiguity (unlike MM/DD/YYYY vs DD/MM/YYYY)
- Works great for monthly business data
- Supported natively by this forecasting app

## Sample CSV File

### Correct YYYYMM Format:
```csv
month,sales
202101,12450
202102,13200
202103,14100
202104,13800
202105,15200
202106,16500
202107,14800
202108,13900
202109,15400
202110,16200
202111,17800
202112,19500
```

### Key Requirements:
✅ **DO:**
- Use consistent YYYYMM format (202301, 202302, etc.)
- Ensure no gaps in months (consecutive months)
- Use numeric values for sales/metrics
- Include header row with column names
- Save as CSV file

❌ **DON'T:**
- Mix formats (don't use both 202301 and 2023-01 in same file)
- Skip months in the sequence
- Use text in the month column
- Include extra date formatting
- Use Excel date serial numbers

## Converting to YYYYMM Format

### From Excel Dates (e.g., 1/1/2023)

**Method 1: Using Excel Formula**
```excel
=TEXT(A2,"YYYYMM")
```
Then copy and paste as values.

**Method 2: Using Excel Custom Format**
1. Select date column
2. Right-click → Format Cells
3. Choose Custom
4. Enter format: `YYYYMM`
5. Click OK

### From Python (Pandas)

```python
import pandas as pd

# Read your existing file
df = pd.read_csv('your_file.csv')

# Convert date column to YYYYMM format
df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y%m').astype(int)

# Save with only required columns
df[['month', 'sales']].to_csv('formatted_data.csv', index=False)
```

### From Date Strings (e.g., "Jan 2023", "January 2023")

**Using Excel:**
```excel
=TEXT(DATEVALUE("01-"&A2),"YYYYMM")
```

**Using Python:**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')
df['month'] = pd.to_datetime(df['date_column']).dt.strftime('%Y%m').astype(int)
df[['month', 'value']].to_csv('output.csv', index=False)
```

### From MM/YYYY Format

**Using Excel:**
```excel
=TEXT(DATEVALUE("01/"&A2),"YYYYMM")
```

**Using Python:**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')
# If format is MM/YYYY
df['month'] = pd.to_datetime(df['date'], format='%m/%Y').dt.strftime('%Y%m').astype(int)
df.to_csv('output.csv', index=False)
```

## Alternative Supported Formats

While YYYYMM is recommended, the app also supports:

### 1. ISO Date Format (YYYY-MM-DD):
```csv
month,sales
2023-01-01,12450
2023-02-01,13200
2023-03-01,14100
```

### 2. Year-Month (YYYY-MM):
```csv
month,sales
2023-01,12450
2023-02,13200
2023-03,14100
```

### 3. Month/Year (MM/YYYY):
```csv
month,sales
01/2023,12450
02/2023,13200
03/2023,14100
```

## Data Quality Checklist

Before uploading your CSV:

- [ ] Dates are in YYYYMM format (or another consistent format)
- [ ] No missing months in the sequence
- [ ] At least 24-36 months of historical data
- [ ] Numeric values for all metrics (no text, no currency symbols)
- [ ] Only two columns: month and value
- [ ] Header row is present
- [ ] File is saved as CSV (not Excel .xlsx)
- [ ] No blank rows at the beginning or end
- [ ] No special characters in column names

## Common Issues and Solutions

### Issue 1: "Date parsing error"
**Problem:** Mixed date formats in one file
**Solution:** Ensure all dates use the same format (preferably YYYYMM)

**Example of WRONG:**
```csv
month,sales
202301,10000
2023-02,12000    ← Mixed format!
202303,11500
```

**Example of RIGHT:**
```csv
month,sales
202301,10000
202302,12000
202303,11500
```

### Issue 2: "Invalid date value"
**Problem:** Text or special characters in month column
**Solution:** Remove any text, keep only numeric YYYYMM values

**Wrong:**
```csv
month,sales
Jan-2023,10000    ← Text format!
```

**Right:**
```csv
month,sales
202301,10000
```

### Issue 3: "Insufficient data"
**Problem:** Too few months
**Solution:** Ensure you have at least 24 months of data for reliable forecasting

### Issue 4: "Missing values"
**Problem:** Gaps in monthly sequence
**Solution:** Fill in missing months or remove gaps

**Wrong:**
```csv
month,sales
202301,10000
202302,12000
202305,11500    ← Missing 202303 and 202304!
```

**Right:**
```csv
month,sales
202301,10000
202302,12000
202303,11200
202304,11800
202305,11500
```

## Example Datasets

### E-commerce Sales (YYYYMM format):
```csv
month,sales
202001,45000
202002,48000
202003,52000
202004,49000
202005,51000
202006,54000
202007,52000
202008,50000
202009,53000
202010,56000
202011,58000
202012,61000
```

### Inventory Levels:
```csv
month,inventory
202101,1250
202102,1180
202103,1340
202104,1290
202105,1410
202106,1380
```

### Revenue Forecasting:
```csv
month,revenue
202001,125000
202002,132000
202003,128000
202004,135000
202005,142000
202006,138000
```

## Quick Conversion Script

Save this as `convert_to_yyyymm.py`:

```python
import pandas as pd
import sys

def convert_to_yyyymm(input_file, date_column, value_column, output_file='output_yyyymm.csv'):
    """
    Convert any date format to YYYYMM format
    
    Usage:
        python convert_to_yyyymm.py input.csv date_col value_col
    """
    # Read the file
    df = pd.read_csv(input_file)
    
    # Convert date to YYYYMM
    df['month'] = pd.to_datetime(df[date_column]).dt.strftime('%Y%m').astype(int)
    
    # Keep only month and value columns
    result = df[['month', value_column]].copy()
    result.columns = ['month', 'value']
    
    # Save
    result.to_csv(output_file, index=False)
    print(f"✅ Converted {len(result)} rows")
    print(f"📁 Saved to: {output_file}")
    print(f"📅 Date range: {result['month'].min()} to {result['month'].max()}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_to_yyyymm.py input.csv date_column value_column")
        sys.exit(1)
    
    convert_to_yyyymm(sys.argv[1], sys.argv[2], sys.argv[3])
```

**Usage:**
```bash
python convert_to_yyyymm.py mydata.csv date sales
```

## Best Practices

1. **Consistency**: Always use the same date format in your file
2. **Completeness**: Don't skip months - fill gaps with estimated values if needed
3. **Accuracy**: Double-check your data before upload
4. **Simplicity**: YYYYMM is simpler and less error-prone than full date formats
5. **Volume**: More data = better forecasts (aim for 36+ months)
6. **Documentation**: Keep notes about data sources and transformations

## Validation Tips

Before uploading, validate your data:

**In Excel:**
1. Sort by month column - should be sequential
2. Check for duplicates
3. Verify no blank cells
4. Confirm all months are present

**In Python:**
```python
import pandas as pd

df = pd.read_csv('your_file.csv')

# Check for gaps
df['month_dt'] = pd.to_datetime(df['month'].astype(str), format='%Y%m')
date_range = pd.date_range(start=df['month_dt'].min(), end=df['month_dt'].max(), freq='MS')
missing = set(date_range) - set(df['month_dt'])
if missing:
    print(f"⚠️ Missing months: {missing}")
else:
    print("✅ No gaps in data")

# Check for duplicates
dupes = df['month'].duplicated().sum()
if dupes > 0:
    print(f"⚠️ Found {dupes} duplicate months")
else:
    print("✅ No duplicates")
```

## Need Help?

If you're having trouble formatting your data:
1. Check the `sample_data.csv` file included with the app
2. Use the conversion scripts provided above
3. Start with a small sample (12 months) to test the format
4. Verify in a text editor before uploading
5. Review error messages in the app

---

**Format your data in YYYYMM and start forecasting! 📊**
