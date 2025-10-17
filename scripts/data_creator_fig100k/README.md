# Fig100k Dataset Processor

This tool converts the fig100k dataset to JSON format similar to the output of `azure_parallel_processor.py`, used for training multimodal reasoning models.

## Features

- 🔄 **Parallel Processing**: Uses multi-threading to process large amounts of data in parallel
- 🤖 **GPT-4o Integration**: Uses Azure OpenAI GPT-4o to analyze image content
- 📊 **Standard Format**: Outputs JSON format compatible with azure_parallel_processor.py
- 🎯 **Intelligent Analysis**: Uses caption as diagram description, GPT-4o analysis as thinking content
- 🔐 **Secure Configuration**: Uses environment variables for API credentials

## Input Format

Fig100k dataset JSON format:
```json
[
  {
    "caption": "Image title/description",
    "context": "Context information (optional)",
    "image_path": "/path/to/image.png"
  }
]
```

## Output Format

### Training Data Format
```json
[
  {
    "stage2_input": {
      "diagram_description_short": "Short diagram description",
      "diagram_description_long": "Detailed diagram description"
    },
    "stage2_output": {
      "thinking_short": "Short thinking process",
      "thinking_long": "Detailed thinking process"
    },
    "metadata": {
      "source": "fig100k",
      "original_caption": "Original caption",
      "original_context": "Original context",
      "image_path": "Image path",
      "processed_time": "Processing time"
    }
  }
]
```

### Judge Data Format
```json
[
  {
    "diagram_description_short": "Short diagram description",
    "diagram_description_long": "Detailed diagram description", 
    "thinking_short": "Short thinking process",
    "thinking_long": "Detailed thinking process",
    "quality_score": 1.0,
    "metadata": {
      "source": "fig100k",
      "original_caption": "Original caption",
      "original_context": "Original context",
      "image_path": "Image path",
      "processed_time": "Processing time"
    }
  }
]
```

## Setup

### 1. Install Dependencies
```bash
pip install requests python-dotenv
```

### 2. Configure Environment Variables
Copy the environment template and fill in your Azure OpenAI credentials:

```bash
cp env_template.txt .env
```

Edit the `.env` file with your actual values:
```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Usage

### Basic Usage
```bash
python fig100k_processor.py \
  --input_json /path/to/paper2fig_train.json \
  --output_dir /path/to/output \
  --workers 4
```

### Parameter Description
- `--input_json, -i`: Input fig100k JSON file path
- `--output_dir, -o`: Output directory
- `--workers, -w`: Number of parallel worker threads (default 4)
- `--max_items, -m`: Maximum number of items to process (for testing)

### Testing Usage
```bash
# Test the processor
python test_fig100k_processor.py

# Process small amount of data for testing
python fig100k_processor.py \
  --input_json /path/to/paper2fig_train.json \
  --output_dir test_output \
  --workers 2 \
  --max_items 10
```

## Output Files

After processing is complete, the output directory will contain:

1. **fig100k_training_data.json**: Training data
2. **fig100k_judge_data.json**: Judge data  
3. **fig100k_processing_report.json**: Processing report
4. **fig100k_processor.log**: Processing log

## Processing Flow

1. **Load Data**: Read fig100k JSON file
2. **Parallel Processing**: Use multi-threading to process each data item
3. **Image Analysis**: Use GPT-4o to analyze image content
4. **Format Conversion**: Convert to standard training/judge format
5. **Save Results**: Output JSON files and reports

## Configuration Requirements

### Azure OpenAI Configuration
The script loads Azure OpenAI parameters from environment variables:
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name (default: gpt-4o)
- `AZURE_OPENAI_API_VERSION`: API version (default: 2024-12-01-preview)

### Dependencies
```bash
pip install requests python-dotenv
```

## Notes

1. **API Limits**: Script includes rate limiting mechanism to avoid excessive API calls
2. **Error Handling**: Includes retry mechanism and detailed error logging
3. **Memory Management**: Suitable for processing large amounts of data, supports parallel processing
4. **Image Paths**: Ensure image file paths are correct and accessible
5. **Security**: API keys are loaded from environment variables, not hardcoded

## Examples

### Process Complete Dataset
```bash
python fig100k_processor.py \
  --input_json /blob/yasheng/paper2fig_train.json \
  --output_dir ./fig100k_output \
  --workers 8
```

### Process Test Data
```bash
python fig100k_processor.py \
  --input_json /blob/yasheng/paper2fig_train.json \
  --output_dir ./test_output \
  --workers 2 \
  --max_items 100
```

## Troubleshooting

### Common Issues

1. **Image File Not Found**
   - Check if image_path is correct
   - Ensure image files are accessible

2. **API Call Failed**
   - Check network connection
   - Verify Azure OpenAI configuration in .env file
   - Check log files for detailed errors

3. **Out of Memory**
   - Reduce number of workers
   - Use max_items to limit processing quantity

4. **Missing API Key**
   - Ensure .env file exists and contains AZURE_OPENAI_API_KEY
   - Check that the API key is valid and has proper permissions

### View Logs
```bash
tail -f fig100k_processor.log
```

### Environment Variable Debug
```bash
# Check if environment variables are loaded correctly
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', bool(os.getenv('AZURE_OPENAI_API_KEY')))"
```
