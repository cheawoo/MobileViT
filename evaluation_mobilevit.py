import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import time

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    model = timm.models.mobilevit_s(pretrained=True).to(device)
    # model = timm.models.mobilevit_xs(pretrained=True).to(device)
    # model = timm.models.mobilevit_xxs(pretrained=True).to(device)
    model = timm.models.mobilenetv2_100(pretrained=True).to(device)
    
    # transform = transforms.Compose(
    #     [transforms.Resize(256),
    #      transforms.CenterCrop(224),
    #      transforms.ToTensor(),
    #      normalize,
    #      ])
    
    # MobileViT 전용 transformer
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    test_set = torchvision.datasets.ImageNet(root="C:/Users/MELLONA/Desktop/datasets", transform=transform, split='val')
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=8)
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_correct_top1 = 0.0
    total_correct_top5 = 0.0
    inf_time = 0.0
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):

            images = images.to(device)  # [100, 3, 224, 224]
            labels = labels.to(device)  # [100]
            start_time = time.time()
            outputs = model(images.cuda())
            end_time = time.time()
            inf_time += end_time - start_time

            # ------------------------------------------------------------------------------
            # rank 1
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            # ------------------------------------------------------------------------------
            # rank 5
            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

            # ------------------------------------------------------------------------------
            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()
            total_correct_top1 += correct_top1 / total * 100
            total_correct_top5 += correct_top5 / total * 100
            print("Step : {} / {}".format(idx + 1, len(test_set) / int(labels.size(0))))
            print("Top-1 percentage : {0:.2f}%".format(correct_top1 / total * 100))
            print("Top-5 percentage : {0:.2f}%".format(correct_top5 / total * 100))
            
    print("\n")
    # print("Model name : mobilevit-small")
    # print("Model name : mobilevit-x-small")
    # print("Model name : mobilevit-xx-small")
    # print("Model name : mobilenetv2")
    print("Total Inf. time: {:0.2f}ms".format(inf_time * 1000)) # record
    print("Top-1 percentage average : {0:.2f}%".format(total_correct_top1 / len(test_loader)))
    print("Top-5 percentage average : {0:.2f}%".format(total_correct_top5 / len(test_loader)))