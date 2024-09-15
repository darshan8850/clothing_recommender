# Clothing Recommender with Multi-Modal RAG

This project demonstrates how to build a dress recommender system using multi-modal Retrieval Augmented Generation (RAG) techniques, combining LLaVA, LangChain, and GPT-4 Vision.

## Overview

The dress recommender suggests outfits based on natural language input specifying type, style, or occasion. It uses a combination of image processing, text embedding, and language models to provide personalized recommendations.

## Features

- Image description generation using LLaVA
- Text embedding and retrieval using sentence transformers
- Multi-vector retrieval with Chroma vector database
- Final recommendation generation using GPT-4 Vision

## Requirements

- Python 3.7+
- CUDA-capable GPU (for LLaVA image processing)
- OpenAI API key (for GPT-4 Vision)

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up LLaVA:
   ```
   cd llama.cpp/
   mkdir build
   cd build 
   cmake .. -DLLAMA_CUBLAS=ON
   cmake --build . --config Release
   ```

## Usage

1. Prepare your dress image dataset
2. Generate image descriptions using LLaVA
3. Create the retriever index
4. Run the recommendation pipeline

For detailed usage instructions, refer to the provided Jupyter notebook.

## Components

1. **Image Representation**: Uses LLaVA to generate textual descriptions of dress images.
2. **Retriever**: Employs sentence transformers and Chroma vector database for efficient image retrieval based on text queries.
3. **LLM Chaining**: Utilizes LangChain to combine retriever results with GPT-4 Vision for final recommendations.

## Performance

- Image description generation: ~9 hours for 1000 images on NVIDIA T4 GPU
- Retrieval: Fast and accurate using sentence transformers
- Final recommendation: ~1 cent per query using GPT-4 Vision

## Future Improvements

- Experiment with GPT-4V for image description generation
- Try different embedding models for retrieval
- Optimize the number of retrieved images sent to the LLM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- LLaVA project
- LangChain library
- OpenAI for GPT-4 Vision API
