""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os
import time

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

from utils.utils import *


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        
        # Set device and precision (support CUDA/MPS/CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Use float16 on CUDA, float32 on MPS/CPU for stability
        target_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model.to(self.device)
        self.model.to(dtype=target_dtype)
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)
        
        Args:
            prompt: Text prompt or list of prompts to guide the model
            image: PIL Image or list of PIL Images to process
            
        Returns:
            Generated text or list of texts from the model
        """
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        # Prepare image (match model dtype/device; silence HF legacy warning when supported)
        try:
            batch_inputs = self.processor(images, return_tensors="pt", padding=True, legacy=False)
        except TypeError:
            batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        model_dtype = next(self.model.parameters()).dtype
        batch_pixel_values = batch_inputs.pixel_values.to(self.device, dtype=model_dtype)
        
        # Prepare prompt
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt"
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0
        )
        
        # Process output
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        # Clean prompt text from output
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
            
        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results


def process_document(document_path, model, save_dir, max_batch_size=None):
    """Parse documents with two stages - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()
    
    if file_ext == '.pdf':
        # Process PDF file
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")
        
        all_results = []
        
        # Process each page
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            start_time = time.perf_counter()
            
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # Process this page (don't save individual page results)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)
            elapsed_s = time.perf_counter() - start_time
            print(f"Finished page {page_idx + 1}/{len(images)} in {elapsed_s:.2f}s")
        
        # Save combined results for multi-page PDF
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        
        return combined_json_path, all_results
    
    else:
        # Process regular image file
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True):
    """Process a single image (either from file or converted from PDF page)
    
    Args:
        image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output file
        max_batch_size: Maximum batch size for processing
        save_individual: Whether to save individual results (False for PDF pages)
        
    Returns:
        Tuple of (json_path, recognition_results)
    """
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)

    # Stage 2: Element-level content parsing
    padded_image, dims = prepare_image(image)
    recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        # Create a dummy image path for save_outputs function
        dummy_image_path = f"{image_name}.jpg"  # Extension doesn't matter, only basename is used
        json_path = save_outputs(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    # Store text and table elements separately
    text_elements = []  # Text elements
    table_elements = []  # Table elements
    figure_results = []  # Image elements (no processing needed)
    previous_box = None
    reading_order = 0

    # Collect elements to process and group by type
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    if save_dir:
                        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                        figure_text = f"![Figure](figures/{figure_filename})"
                        figure_path = f"figures/{figure_filename}"
                    else:
                        # When save_dir is None, just mark it as a figure without saving
                        figure_text = "[Figure]"
                        figure_path = None
                    
                    # For figure regions, store relative path or placeholder
                    figure_results.append(
                        {
                            "label": label,
                            "text": figure_text,
                            "figure_path": figure_path,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # Prepare element for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    
                    # Group by type
                    if label == "tab":
                        table_elements.append(element_info)
                    else:  # Text elements
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Initialize results list
    recognition_results = figure_results.copy()
    
    # Process text elements (in batches)
    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)
    
    # Process table elements (in batches)
    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []
    
    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)
        
        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)
        
        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
    
    return results


def batch_detect_all_elements(image_paths, model, max_batch_size=16):
    """
    CUDA-optimized batch processing: detect all elements from multiple images in parallel.
    
    This function leverages the existing chat() batching capability to maximize GPU utilization:
    1. Batch layout detection across all images
    2. Batch element recognition across all detected elements
    
    Args:
        image_paths: List of image file paths
        model: DOLPHIN model instance (with batching support via chat())
        max_batch_size: Maximum elements per batch for GPU memory optimization
        
    Returns:
        Dict mapping image_path -> list of detected elements 
        Each element: {"label": str, "bbox": [x1,y1,x2,y2], "text": str, "reading_order": int}
    """
    if not image_paths:
        return {}
        
    print(f"Batch processing {len(image_paths)} images with max_batch_size={max_batch_size}")
    
    # Load all images and prepare them
    images = []
    image_dims = []
    padded_images = []
    
    for img_path in image_paths:
        pil_image = Image.open(img_path).convert("RGB")
        images.append(pil_image)
        padded_img, dims = prepare_image(pil_image)
        padded_images.append(padded_img)
        image_dims.append(dims)
    
    # Stage 1: Batch layout detection for ALL images in one GPU call
    layout_prompts = ["Parse the reading order of this document."] * len(images)
    start_time = time.perf_counter()
    layout_outputs = model.chat(layout_prompts, images)
    stage1_time = time.perf_counter() - start_time
    print(f"Stage 1 (Layout): {len(images)} images processed in {stage1_time:.2f}s")
    
    # Stage 2: Extract all elements across all images and batch process them
    all_text_elements = []
    all_table_elements = []  
    results_by_image = {}
    
    # Process layout results and collect all elements
    for idx, (img_path, layout_output, padded_image, dims) in enumerate(zip(image_paths, layout_outputs, padded_images, image_dims)):
        layout_results = parse_layout_string(layout_output)
        figure_results = []
        previous_box = None
        reading_order = 0
        
        for bbox, label in layout_results:
            try:
                x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                    bbox, padded_image, dims, previous_box
                )
                
                cropped = padded_image[y1:y2, x1:x2]
                if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    if label == "fig":
                        # Handle figures immediately (no GPU processing needed)
                        figure_results.append({
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": "[Figure]",
                            "reading_order": reading_order,
                        })
                    else:
                        # Collect for batch processing
                        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                            "image_path": img_path,
                        }
                        
                        if label == "tab":
                            all_table_elements.append(element_info)
                        else:
                            all_text_elements.append(element_info)
                
                reading_order += 1
                
            except Exception as e:
                print(f"Error processing element in {img_path}: {str(e)}")
                continue
        
        results_by_image[img_path] = figure_results
    
    # Batch process all text elements across ALL images
    if all_text_elements:
        start_time = time.perf_counter()
        crops_list = [elem["crop"] for elem in all_text_elements]
        prompts_list = ["Read text in the image."] * len(crops_list)
        
        # Process in chunks to manage GPU memory
        for i in range(0, len(all_text_elements), max_batch_size):
            end_idx = min(i + max_batch_size, len(all_text_elements))
            batch_crops = crops_list[i:end_idx]
            batch_prompts = prompts_list[i:end_idx]
            batch_elements = all_text_elements[i:end_idx]
            
            # Single GPU batch call
            batch_results = model.chat(batch_prompts, batch_crops)
            
            # Distribute results back to their images
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                img_path = elem["image_path"]
                results_by_image[img_path].append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                })
        
        text_time = time.perf_counter() - start_time
        print(f"Stage 2a (Text): {len(all_text_elements)} elements processed in {text_time:.2f}s")
    
    # Batch process all table elements across ALL images  
    if all_table_elements:
        start_time = time.perf_counter()
        crops_list = [elem["crop"] for elem in all_table_elements]
        prompts_list = ["Parse the table in the image."] * len(crops_list)
        
        # Process in chunks to manage GPU memory
        for i in range(0, len(all_table_elements), max_batch_size):
            end_idx = min(i + max_batch_size, len(all_table_elements))
            batch_crops = crops_list[i:end_idx]
            batch_prompts = prompts_list[i:end_idx]
            batch_elements = all_table_elements[i:end_idx]
            
            # Single GPU batch call
            batch_results = model.chat(batch_prompts, batch_crops)
            
            # Distribute results back to their images
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                img_path = elem["image_path"]
                results_by_image[img_path].append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                })
        
        table_time = time.perf_counter() - start_time
        print(f"Stage 2b (Tables): {len(all_table_elements)} elements processed in {table_time:.2f}s")
    
    # Sort elements within each image by reading order
    for img_path in results_by_image:
        results_by_image[img_path].sort(key=lambda x: x.get("reading_order", 0))
    
    total_elements = sum(len(results) for results in results_by_image.values())
    print(f"Completed: {total_elements} total elements from {len(image_paths)} images")
    
    return results_by_image


def batch_detect_all_elements_pil(images, image_names, model, max_batch_size=16):
    """
    PIL image version for Modal deployment: detect all elements from PIL images in parallel.
    
    Args:
        images: List of PIL Image objects
        image_names: List of image names/identifiers
        model: DOLPHIN model instance (with batching support via chat())
        max_batch_size: Maximum elements per batch for GPU memory optimization
        
    Returns:
        Dict mapping image_name -> list of detected elements 
        Each element: {"label": str, "bbox": [x1,y1,x2,y2], "text": str, "reading_order": int}
    """
    if not images:
        return {}
        
    print(f"Batch processing {len(images)} PIL images with max_batch_size={max_batch_size}")
    
    # Prepare all images
    image_dims = []
    padded_images = []
    
    for pil_image in images:
        padded_img, dims = prepare_image(pil_image)
        padded_images.append(padded_img)
        image_dims.append(dims)
    
    # Stage 1: Batch layout detection for ALL images in one GPU call
    layout_prompts = ["Parse the reading order of this document."] * len(images)
    start_time = time.perf_counter()
    layout_outputs = model.chat(layout_prompts, images)
    stage1_time = time.perf_counter() - start_time
    print(f"Stage 1 (Layout): {len(images)} images processed in {stage1_time:.2f}s")
    
    # Stage 2: Extract all elements across all images and batch process them
    all_text_elements = []
    all_table_elements = []  
    results_by_image = {}
    
    # Process layout results and collect all elements
    for idx, (img_name, layout_output, padded_image, dims) in enumerate(zip(image_names, layout_outputs, padded_images, image_dims)):
        layout_results = parse_layout_string(layout_output)
        figure_results = []
        previous_box = None
        reading_order = 0
        
        for bbox, label in layout_results:
            try:
                x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                    bbox, padded_image, dims, previous_box
                )
                
                cropped = padded_image[y1:y2, x1:x2]
                if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    if label == "fig":
                        # Handle figures immediately (no GPU processing needed)
                        figure_results.append({
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": "[Figure]",
                            "reading_order": reading_order,
                        })
                    else:
                        # Collect for batch processing
                        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                            "image_name": img_name,
                        }
                        
                        if label == "tab":
                            all_table_elements.append(element_info)
                        else:
                            all_text_elements.append(element_info)
                
                reading_order += 1
                
            except Exception as e:
                print(f"Error processing element in {img_name}: {str(e)}")
                continue
        
        results_by_image[img_name] = figure_results
    
    # Batch process all text elements across ALL images
    if all_text_elements:
        start_time = time.perf_counter()
        crops_list = [elem["crop"] for elem in all_text_elements]
        prompts_list = ["Read text in the image."] * len(crops_list)
        
        # Process in chunks to manage GPU memory
        for i in range(0, len(all_text_elements), max_batch_size):
            end_idx = min(i + max_batch_size, len(all_text_elements))
            batch_crops = crops_list[i:end_idx]
            batch_prompts = prompts_list[i:end_idx]
            batch_elements = all_text_elements[i:end_idx]
            
            # Single GPU batch call
            batch_results = model.chat(batch_prompts, batch_crops)
            
            # Distribute results back to their images
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                img_name = elem["image_name"]
                results_by_image[img_name].append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                })
        
        text_time = time.perf_counter() - start_time
        print(f"Stage 2a (Text): {len(all_text_elements)} elements processed in {text_time:.2f}s")
    
    # Batch process all table elements across ALL images  
    if all_table_elements:
        start_time = time.perf_counter()
        crops_list = [elem["crop"] for elem in all_table_elements]
        prompts_list = ["Parse the table in the image."] * len(crops_list)
        
        # Process in chunks to manage GPU memory
        for i in range(0, len(all_table_elements), max_batch_size):
            end_idx = min(i + max_batch_size, len(all_table_elements))
            batch_crops = crops_list[i:end_idx]
            batch_prompts = prompts_list[i:end_idx]
            batch_elements = all_table_elements[i:end_idx]
            
            # Single GPU batch call
            batch_results = model.chat(batch_prompts, batch_crops)
            
            # Distribute results back to their images
            for j, result in enumerate(batch_results):
                elem = batch_elements[j]
                img_name = elem["image_name"]
                results_by_image[img_name].append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                })
        
        table_time = time.perf_counter() - start_time
        print(f"Stage 2b (Tables): {len(all_table_elements)} elements processed in {table_time:.2f}s")
    
    # Sort elements within each image by reading order
    for img_name in results_by_image:
        results_by_image[img_name].sort(key=lambda x: x.get("reading_order", 0))
    
    total_elements = sum(len(results) for results in results_by_image.values())
    print(f"Completed: {total_elements} total elements from {len(images)} images")
    
    return results_by_image


def main():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image/PDF or directory of files")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum number of document elements to parse in a single batch (default: 16)",
    )
    args = parser.parse_args()

    # Load Model
    model = DOLPHIN(args.model_path)

    # Collect Document Files (images and PDFs)
    if os.path.isdir(args.input_path):
        # Support both image and PDF files
        file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        
        document_files = []
        for ext in file_extensions:
            document_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        document_files = sorted(document_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        
        # Check if it's a supported file type
        file_ext = os.path.splitext(args.input_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        
        if file_ext not in supported_exts:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported_exts}")
        
        document_files = [args.input_path]

    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    setup_output_dirs(save_dir)

    total_samples = len(document_files)
    print(f"\nTotal files to process: {total_samples}")

    # Process All Document Files
    for file_path in document_files:
        print(f"\nProcessing {file_path}")
        try:
            json_path, recognition_results = process_document(
                document_path=file_path,
                model=model,
                save_dir=save_dir,
                max_batch_size=args.max_batch_size,
            )

            print(f"Processing completed. Results saved to {save_dir}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
