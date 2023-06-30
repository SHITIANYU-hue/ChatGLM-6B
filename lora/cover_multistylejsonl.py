import argparse
import json
from tqdm import tqdm

def format_example_multi(example: dict,style:str) -> dict:
    if style =='keai':
        try:
            context = f"Instruction: 假设你是AI虚拟人，根据如下人设：'{example['input']}'，用可爱的风格回复该问题：{example['instruction']}，你的回复：\n"
            target=example["output"]['keai']
            return {"context": context, "target": target}
        except:
            pass
    if style =='gufeng':
        try:
            context = f"Instruction: 假设你是AI虚拟人，根据如下人设：'{example['input']}'，用古风的风格回复该问题：{example['instruction']}，你的回复：\n"
            target=example["output"]['gufeng']
            return {"context": context, "target": target}
        except:
            pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/amax/code/ChatGLM-6B/ptuning/character/multi2/data_multi.json")
    parser.add_argument("--save_path", type=str, default="/home/amax/code/ChatGLM-6B/ptuning/character/multi2/data_multi.jsonl")

    args = parser.parse_args()
    with open(args.data_path) as f:
        examples = json.load(f)

    styles=['keai','gufeng']
    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            for style in styles:
                f.write(json.dumps(format_example_multi(example,style),ensure_ascii=False)+'\n')

if __name__ == "__main__":
    main()
