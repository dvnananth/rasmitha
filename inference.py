import os

import torch
import wandb
from huggingface_hub import HfApi, login
from tqdm import tqdm

try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def main() -> None:
    transformers_available = TRANSFORMERS_AVAILABLE

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT", "TA duty demo project")

    use_wandb = bool(wandb_api_key)

    if use_wandb:
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=wandb_project,
            name="hindi-to-english-translation",
            config={
                "model_name": "Helsinki-NLP/opus-mt-hi-en",
                "device": "cpu",
            },
        )
    else:
        print("[INFO] WANDB_API_KEY not set. Running in test mode without W&B logging.")

    model_name = "Helsinki-NLP/opus-mt-hi-en"
    upload_to_hf = os.environ.get("UPLOAD_TO_HF", "false").lower() == "true"
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID", "DuckyDuck123/hindi-en-demo-model")

    print("\n[LOADING] Loading Hindi-to-English translation model from Hugging Face...")
    
    translated_texts = []
    
    if transformers_available:
        with tqdm(total=2, desc="Model Loading", unit="step") as pbar:
            print(f"[INFO] Using model: {model_name}")
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                pbar.update(1)
                model = MarianMTModel.from_pretrained(model_name)
                pbar.update(1)
                model.eval()
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
                transformers_available = False

    if upload_to_hf:
        if not transformers_available:
            print("[WARNING] Skipping Hugging Face upload because model is not available.")
        elif not hf_token:
            print("[WARNING] Skipping Hugging Face upload because HF_TOKEN is not set.")
        else:
            print(f"\n[UPLOAD] Uploading model and tokenizer to Hugging Face repo: {hf_repo_id}")
            with tqdm(total=3, desc="HF Upload", unit="step") as pbar:
                login(token=hf_token)
                pbar.update(1)
                api = HfApi()
                api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
                pbar.update(1)
                tokenizer.push_to_hub(hf_repo_id)
                model.push_to_hub(hf_repo_id)
                pbar.update(1)
            print(f"[UPLOAD] Completed: https://huggingface.co/{hf_repo_id}")

    texts = [
        "नमस्ते, यह एक हिंदी से अंग्रेजी अनुवाद का डेमो है।",
        "मशीन लर्निंग बहुत उपयोगी है।",
        "गिटहब एक्शन्स का उपयोग करके हम स्वचालित वर्कफ्लो बना सकते हैं।",
    ]

    print("\n[TRANSLATION] Processing Hindi texts and translating to English...")
    
    if transformers_available:
        with tqdm(total=len(texts), desc="Translating", unit="text") as pbar:
            for hindi_text in texts:
                try:
                    inputs = tokenizer(hindi_text, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        translated = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_texts.append(translated_text)
                except Exception as e:
                    print(f"[ERROR] Translation failed: {e}")
                    translated_texts.append(f"[Translation error]")
                pbar.update(1)
    else:
        # Fallback to mock translations if model not available
        print("[INFO] Using mock translations (model not available)")
        mock_translations = [
            "Hello, this is a demo of Hindi to English translation.",
            "Machine learning is very useful.",
            "Using GitHub Actions, we can create automated workflows.",
        ]
        with tqdm(total=len(texts), desc="Translating", unit="text") as pbar:
            for _ in texts:
                translated_texts.append(mock_translations[len(translated_texts)])
                pbar.update(1)

    if use_wandb:
        print("\n[LOGGING] Preparing W&B logging...")
        with tqdm(total=len(texts) + 1, desc="W&B Upload", unit="row") as pbar:
            table = wandb.Table(columns=["hindi_text", "english_translation"])
            for hindi, english in zip(texts, translated_texts):
                table.add_data(hindi, english)
                pbar.update(1)

            wandb.log(
                {
                    "translation_table": table,
                    "total_translations": len(texts),
                    "model_used": "actual" if transformers_available else "mock",
                }
            )
            pbar.update(1)

    print("\n" + "="*60)
    print("✓ Translation completed successfully!")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Inference Type: {'ACTUAL MODEL' if transformers_available else 'MOCK (Model not available)'}")
    print(f"Total translations: {len(translated_texts)}")
    print("\nTranslations:")
    for i, (hindi, english) in enumerate(zip(texts, translated_texts), 1):
        print(f"\n[{i}] Hindi: {hindi}")
        print(f"    English: {english}")
    if use_wandb:
        print("\n✓ Results logged to W&B project:", wandb_project)
    else:
        print("\nℹ W&B logging disabled (no API key provided).")
    print("="*60 + "\n")

    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()