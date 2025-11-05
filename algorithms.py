from .algorithms import Algorithm
from .networks import EEGFeaturizer, EEGClassifier
import torch
import torch.nn as nn
import torch.optim as optim


class EEGDANN(Algorithm):
    """脑电领域的域对抗神经网络"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = EEGFeaturizer(input_shape, hparams)
        feat_dim = self.featurizer.n_outputs * self.featurizer.final_len

        self.classifier = EEGClassifier(feat_dim, num_classes)
        self.domain_discriminator = nn.Linear(feat_dim, num_domains)

        # 域对抗梯度反转层
        self.grl = GradientReversalLayer()
        self.optimizer = optim.Adam(
            list(self.featurizer.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_discriminator.parameters()),
            lr=hparams["lr"]
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((len(x),), i, device=x.device)
            for i, (x, y) in enumerate(minibatches)
        ])

        # 特征提取
        feats = self.featurizer(all_x)

        # 分类损失
        class_logits = self.classifier(feats)
        class_loss = nn.CrossEntropyLoss()(class_logits, all_y)

        # 域鉴别损失（梯度反转）
        domain_logits = self.domain_discriminator(self.grl(feats))
        domain_loss = nn.CrossEntropyLoss()(domain_logits, all_d)

        # 总损失
        loss = class_loss + self.hparams["domain_loss_weight"] * domain_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "class_loss": class_loss.item(),
            "domain_loss": domain_loss.item()
        }

    def predict(self, x):
        feats = self.featurizer(x)
        return self.classifier(feats)


class GradientReversalLayer(nn.Module):
    """梯度反转层（DANN核心）"""

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -grad_output