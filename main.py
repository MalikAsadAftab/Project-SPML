# -*- coding: utf-8 -*-
import argparse
from attacks.Attack0.attack0_GraphSAGE import attack0_GraphSAGE
from attacks.Attack0.attack0_GAT_Link import attack0_GAT_Link
from attacks.Attack1.attack_1 import attack1
from attacks.Attack2.attack_2 import attack2
from attacks.Attack3.attack_3 import attack3

def main():
    parser = argparse.ArgumentParser(description="Run specified attack type on a given dataset.")

    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--attack_type', type=int, default=0,
                        help="Integer ID of attack type.")
    parser.add_argument('--dataset', type=str, default='citeseer',
                        help="Dataset for the target model: (cora, citeseer, pubmed).")
    parser.add_argument('--attack_node', type=float, default=0.25,
                        help='Proportion of the attack nodes.')
    parser.add_argument('--shadow_dataset_size', type=float, default=1,
                        help='Size of the shadow datasets.')

    args = parser.parse_args()

    print(f"Selected attack type: {args.attack_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Attack node proportion: {args.attack_node * 100}%")

    if args.gpu:
        print(f"Using GPU ID: {args.gpu}")
    else:
        print("Using CPU")

    attack_functions = {
        0: attack0_GraphSAGE,
        1: attack1,
        2: attack2,
        3: attack3,
        4: attack0_GAT_Link
    }

    if args.attack_type in attack_functions:
        attack_function = attack_functions[args.attack_type]
        print(f"Running {attack_function.__name__}...")
        attack_function(args.dataset, args.attack_node, args.gpu)
    else:
        print(f"Invalid attack_type: {args.attack_type}. Please provide a valid attack type.")
        print("Valid attack types are:")
        for key in attack_functions.keys():
            print(f"{key}: {attack_functions[key].__name__}")


if __name__ == '__main__':
    main()