# %% [markdown]
# # Brain MRI: LRP + Grad-CAM on the VGG16 checkpoint trained by 05_lrp.py
# Run in Colab (GPU) from inside the xai-series folder, after data/brain_mri/{training,testing}
# exist and vgg16_brain_mri_baseline.pth has been placed in the working directory.
# Preprocessing is deliberately identical to 05_lrp.py (Resize 255x255, ToTensor, NO ImageNet
# normalisation) because the checkpoint was trained that way.

# %% Imports and setup
import os, copy, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.models as models, torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

SEED = 2026
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__, "| torchvision", torchvision.__version__, "| numpy", np.__version__)

TRAIN_ROOT = "data/brain_mri/training"
TEST_ROOT = "data/brain_mri/testing"
CKPT = "vgg16_brain_mri_baseline.pth"
os.makedirs("figures", exist_ok=True)

# %% Data (test loader is NOT shuffled so indices are stable and reproducible)
tf = transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(TRAIN_ROOT, transform=tf)
test_dataset = torchvision.datasets.ImageFolder(TEST_ROOT, transform=tf)
assert train_dataset.class_to_idx == test_dataset.class_to_idx, "class mappings differ"
class_names = test_dataset.classes
print("class_to_idx:", test_dataset.class_to_idx)
print("train images:", len(train_dataset), "| test images:", len(test_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% Model (same definition as 05_lrp.py) + checkpoint
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = models.vgg16(weights=None)  # weights come from the checkpoint
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, 4)
    def forward(self, x):
        return self.vgg16(x)

model = CNNModel()
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device).eval()   # eval(): disables dropout for prediction and explanations

# %% Full test-set evaluation
all_pred, all_true, all_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        p = F.softmax(model(x.to(device)), dim=1).cpu()
        all_prob.append(p); all_pred.append(p.argmax(1)); all_true.append(y)
pred = torch.cat(all_pred).numpy(); true = torch.cat(all_true).numpy(); prob = torch.cat(all_prob).numpy()
print("Test images evaluated:", len(true), "| accuracy: %.4f" % (pred == true).mean())
print(classification_report(true, pred, target_names=class_names, digits=3))
cm = confusion_matrix(true, pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(xticks_rotation=30, cmap="Blues")
plt.title("Confusion matrix (counts; rows = dataset label, columns = prediction)")
plt.tight_layout(); plt.savefig("figures/confusion_matrix.png", dpi=200); plt.show()

# %% LRP (starter code from 05_lrp.py; changes marked CHANGED)
def new_layer(layer, g):
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer

def dense_to_conv(layers):
    newlayers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers

def get_linear_layer_indices(model):
    offset = len(model.vgg16._modules['features']) + 1
    return [offset + i for i, l in enumerate(model.vgg16._modules['classifier']) if isinstance(l, nn.Linear)]

def lrp_vgg16(model, image, target=None):
    """Returns (relevance HxWx3 numpy, explained class index). image: 3xHxW tensor.
    CHANGED: target class can be chosen (default = predicted class); one-hot built in torch."""
    image = torch.unsqueeze(image, 0).to(device)
    layers = list(model.vgg16._modules['features']) + [model.vgg16._modules['avgpool']] \
             + dense_to_conv(list(model.vgg16._modules['classifier']))
    linear_layer_indices = get_linear_layer_indices(model)
    n_layers = len(layers)
    activations = [image] + [None] * n_layers
    with torch.no_grad():
        for layer in range(n_layers):
            if layer in linear_layer_indices and layer == 32:
                activations[layer] = activations[layer].reshape((1, 512, 7, 7))
            activation = layers[layer].forward(activations[layer])
            if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
                activation = torch.flatten(activation, start_dim=1)
            activations[layer + 1] = activation
    # CHANGED: Step 3 in torch (original raised TypeError under NumPy 2.x)
    logits = activations[-1].detach()
    t = int(logits.argmax(1)) if target is None else int(target)
    one_hot = torch.zeros_like(logits)
    one_hot[0, t] = logits[0, t]
    activations[-1] = one_hot.to(device)

    relevances = [None] * n_layers + [activations[-1]]
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2); current = layers[layer]
        if isinstance(current, (torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.Linear)):
            activations[layer] = activations[layer].data.requires_grad_(True)
            if layer <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9           # LRP-gamma
            if 17 <= layer <= 30: rho = lambda p: p;                         incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data  # LRP-epsilon
            if layer >= 31:       rho = lambda p: p;                         incr = lambda z: z + 1e-9           # LRP-0
            z = incr(new_layer(layers[layer], rho).forward(activations[layer]))
            s = (relevances[layer + 1] / z).data
            (z * s).sum().backward()
            c = activations[layer].grad
            relevances[layer] = (activations[layer] * c).data
        else:
            relevances[layer] = relevances[layer + 1]
    return relevances[0].permute(0, 2, 3, 1).detach().cpu().numpy()[0], t

# %% Grad-CAM (own implementation, no hooks; target layer = last conv block, features[29] = ReLU after conv 28)
def gradcam_vgg16(model, image, target=None):
    x = image.unsqueeze(0).to(device)
    vgg = model.vgg16
    a = vgg.features[:30](x)                       # activations of the last conv layer (after ReLU)
    out = vgg.classifier(torch.flatten(vgg.avgpool(vgg.features[30](a)), 1))
    t = int(out.argmax(1)) if target is None else int(target)
    grads = torch.autograd.grad(out[0, t], a)[0]   # d score_t / d activations (raw logit)
    w = grads.mean(dim=(2, 3), keepdim=True)       # channel weights
    cam = F.relu((w * a).sum(1, keepdim=True))     # negative evidence discarded
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-12), t

# %% Display helpers (signed LRP: symmetric colour scale so white = zero relevance)
def lrp_map(rel):                    # sum over RGB channels, keep sign
    return rel.sum(axis=2)

def show_case(idx, fname):
    x, y = test_dataset[idx]
    p = int(pred[idx]); conf = prob[idx, p]
    rel, t = lrp_vgg16(model, x, target=p)
    cam, _ = gradcam_vgg16(model, x, target=p)
    r = lrp_map(rel); v = np.percentile(np.abs(r), 99.5)
    img = x.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(img); ax[0].set_title("A  Original MRI")
    ax[1].imshow(img.mean(2), cmap="gray")
    im1 = ax[1].imshow(r, cmap="bwr", vmin=-v, vmax=v, alpha=0.6); ax[1].set_title("B  LRP (red +, blue -)")
    ax[2].imshow(img.mean(2), cmap="gray")
    im2 = ax[2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.5); ax[2].set_title("C  Grad-CAM (0-1)")
    for a_ in ax: a_.axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046); plt.colorbar(im2, ax=ax[2], fraction=0.046)
    path = test_dataset.samples[idx][0]
    fig.suptitle(f"{os.path.basename(path)} | label: {class_names[y]} | prediction: {class_names[p]} "
                 f"(softmax {conf:.2f}) | target: {class_names[p]}", fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.show()
    print(path)

# %% Case selection: first correct image of each class + first misclassified images
cases = {}
for c in range(len(class_names)):
    ok = np.where((true == c) & (pred == c))[0]
    if len(ok): cases[f"correct_{class_names[c]}"] = int(ok[0])
wrong = np.where(pred != true)[0]
for k, i in enumerate(wrong[:3]):
    cases[f"error_{k+1}_{class_names[true[i]]}_as_{class_names[pred[i]]}"] = int(i)
print(cases)
for name, idx in cases.items():
    show_case(idx, f"figures/case_{name}.png")

# %% Quantitative bonus: deletion test (remove most-relevant pixels, watch the predicted-class probability fall)
# Masked pixels are set to 0 (black = MRI background); this introduces masking artefacts. It tests
# faithfulness to the MODEL, not diagnostic validity.
def deletion_curve(img, mapping, cls, fracs):
    order = np.argsort(-mapping.ravel())
    out = []
    for f in fracs:
        m = np.ones(mapping.size, dtype=np.float32); m[order[:int(f * mapping.size)]] = 0
        xm = img * torch.from_numpy(m.reshape(mapping.shape))[None]
        with torch.no_grad():
            out.append(F.softmax(model(xm[None].to(device)), 1)[0, cls].item())
    return np.array(out)

def auc(y, x): return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))

fracs = np.linspace(0, 0.5, 11)
ok_idx = np.where(pred == true)[0]
sel = np.random.RandomState(SEED).choice(ok_idx, size=min(30, len(ok_idx)), replace=False)
res = {"LRP (positive)": [], "Grad-CAM": [], "Random": []}
for i in sel:
    x, _ = test_dataset[int(i)]; c = int(pred[i])
    rel, _ = lrp_vgg16(model, x, target=c); cam, _ = gradcam_vgg16(model, x, target=c)
    res["LRP (positive)"].append(deletion_curve(x, np.clip(lrp_map(rel), 0, None), c, fracs))
    res["Grad-CAM"].append(deletion_curve(x, cam, c, fracs))
    res["Random"].append(deletion_curve(x, np.random.RandomState(int(i)).rand(*cam.shape), c, fracs))
plt.figure(figsize=(6, 4))
for k, v in res.items():
    m = np.mean(v, 0); plt.plot(fracs * 100, m, marker="o", label=f"{k} (AUC {auc(m, fracs):.3f})")
plt.xlabel("% of pixels removed (most relevant first)"); plt.ylabel("mean probability of predicted class")
plt.title(f"Deletion test, {len(sel)} correctly classified test images"); plt.legend(); plt.tight_layout()
plt.savefig("figures/deletion_test.png", dpi=200); plt.show()
# Lower AUC = faster drop = the map found pixels the model relies on more.
