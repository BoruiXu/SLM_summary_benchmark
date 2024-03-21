import argparse
import logging
import sys


class ColorizingStreamHandler(logging.StreamHandler):
    color_map = {
        logging.DEBUG: "\033[1;34m",  # Blue
        logging.INFO: "\033[1;32m",   # Green
        logging.WARNING: "\033[1;33m",  # Yellow
        logging.ERROR: "\033[1;31m",  # Red
        logging.CRITICAL: "\033[1;41m"  # White on Red (background)
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        color = self.color_map.get(record.levelno, "\033[0m")  # Reset color
        return color + super().format(record) + "\033[0m"  # Reset color

# 创建日志记录器
eval_logger = logging.getLogger()
eval_logger.setLevel(logging.INFO)

# 创建带有颜色的日志处理程序
handler = ColorizingStreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    "%Y-%m-%d:%H:%M:%S"
)
handler.setFormatter(formatter)

# 将处理程序添加到日志记录器
eval_logger.addHandler(handler)



def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="model type hf, api, hf_chat, api_chat"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        choices=['generate', 'evaluate'],
        metavar="task type",
        help="To set task type one of ['generate', 'evaluate']",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=0,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--fewshot_dataset",
        type=str,
        default=None,
        help="data used for fewshot context",
    )
    
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        metavar="prompt",
        help="The prompt to use for the task.",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default="results.json",
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result will be saved.",
    )
    
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default="do_sample=False,max_new_tokens=200 ",
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )

    return parser

# # # 解析命令行参数
# args = setup_parser().parse_args()

# # # 使用解析结果
# print((args))
