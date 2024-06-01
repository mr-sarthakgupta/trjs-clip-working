import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from '@xenova/transformers';

const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-large-patch14');
const vision_model = await CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-large-patch14');

// Read image and run processor
const image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
const image_inputs = await processor(image);

// Compute embeddings
const { image_embeds } = await vision_model(image_inputs);

console.log(image_embeds['data'][0]);