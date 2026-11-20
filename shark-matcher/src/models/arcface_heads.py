"""ArcFace margin heads and loss layers for metric learning."""

import math

import torch


class ArcMarginProduct(torch.nn.Module):
    """Large-margin arc distance head for metric learning."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """Forward pass computing angular margin logits."""
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # Torch functional symbols are runtime callables despite Pylint inference.
        # pylint: disable-next=not-callable
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input), torch.nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProduct_subcenter(torch.nn.Module):
    """ArcMarginProduct head with multiple sub-centers per class."""
    def __init__(self, in_features, out_features, k=3):
        """Initialize ArcMarginProduct_subcenter with sub-center weight parameters."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        """Initialize sub-center weights uniformly within plus or minus 1/sqrt(in_features)."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        """Forward pass computing max cosine similarity across sub-centers."""
        # Torch functional symbols are runtime callables despite Pylint inference.
        # pylint: disable-next=not-callable
        cosine_all = torch.nn.functional.linear(torch.nn.functional.normalize(features), torch.nn.functional.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcFaceLossAdaptiveMargin(torch.nn.Module):
    """ArcFace loss layer with class-adaptive angular margins."""
    def __init__(self, margins, out_dim, s):
        """Initialize ArcFaceLossAdaptiveMargin with per-class margins and scale factor."""
        super().__init__()
        self.s = s
        self.register_buffer('margins', margins.clone().detach())
        self.out_dim = out_dim
            
    def forward(self, logits, labels):
        """Forward pass applying class-adaptive angular margins to logits."""
        ms = self.margins[labels]
        cos_m = torch.cos(ms)
        sin_m = torch.sin(ms)
        th = torch.cos(math.pi - ms)
        mm = torch.sin(math.pi - ms) * ms
        # Torch functional symbols are runtime callables despite Pylint inference.
        # pylint: disable-next=not-callable
        labels = torch.nn.functional.one_hot(labels, self.out_dim).float()
        cosine = logits
        sine = torch.sqrt(1.0 - cosine * cosine)
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output


class ArcFaceSubCenterDynamic(torch.nn.Module):
    """Sub-center ArcFace head with dynamic adaptive margins."""
    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        k=3,
        margins=None
    ):
        """Initialize ArcFaceSubCenterDynamic head."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.margins = margins
        self.s = s
        
        # Sub-center metric learning
        self.wmetric_classify = ArcMarginProduct_subcenter(self.in_features, self.out_features, k=k)
        
        # Adaptive Margin Layer
        self.warcface_margin = ArcFaceLossAdaptiveMargin(margins=self.margins, 
                                                         out_dim=self.out_features, 
                                                         s=self.s)

    def forward(self, features, labels):
        """Forward pass returning margin-scaled logits for loss computation."""
        logits = self.wmetric_classify(features.float())
        logits = self.warcface_margin(logits, labels)
        return logits
