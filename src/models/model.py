from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoFeatureExtractor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
feat_ext = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(feat_ext, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")

# TODO: This is just a copy of CLIP pre-run code.
#  Should be reformated to a func.