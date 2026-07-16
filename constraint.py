import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        # Dynamically match the device of the input features (e.g., cuda:0)
        current_device = X.device
        
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        
        # Ensure mu calculation is moved to the correct device
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y).to(current_device)
        
        # Ensure constraint matrices M and c are mapped to the active device
        gap_constraint = F.relu(
            torch.mv(self.M.to(current_device), mu) - self.c.to(current_device)
        )
        
        # Ensure self.alpha (which might be a float, int, or tensor) works on the target device
        alpha_t = self.alpha.to(current_device) if isinstance(self.alpha, torch.Tensor) else self.alpha
        
        if self.p_norm == 2:
            cons = alpha_t * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = alpha_t * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons
class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(out[idx_true].mean())
            
        expected_values_list.append(out.mean())
        return torch.stack(expected_values_list)


    # TASK 3 Changes: DUAL-ATTRIBUTE MINIMAX FORWARD PASS
    # Computes demographic parity constraint violations for both gender and race.
    # To prevent expanding the Pareto space, we apply a minimax formulation
    # which dynamically penalizes only the worst-performing attribute.
    def forward(self, X, out, sensitive_gender, sensitive_race=None, y=None):
        # If no second attribute is provided, fall back to single-attribute
        if sensitive_race is None:
            return super(DemographicParityLoss, self).forward(X, out, sensitive_gender)
            
        # Compute individual constraint losses
        loss_gender = super(DemographicParityLoss, self).forward(X, out, sensitive_gender)
        loss_race = super(DemographicParityLoss, self).forward(X, out, sensitive_race)
        
        # Minimax formulation: Dynamically return the worst-violating loss path
        if loss_gender.item() > loss_race.item():
            return loss_gender
        else:
            return loss_race

class AverageTreatmentEffectLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        self.sensitive_classes = sensitive_classes
        self.y_classes = [1]  # only consider positive outcome
        self.n_class = len(sensitive_classes)
        self.n_y_class = len(self.y_classes)
        # Bug B: super(EqualOpportunityLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        # Bug B Fix
        super(AverageTreatmentEffectLoss, self).__init__(n_class=self.n_class, alpha=alpha, p_norm=p_norm)
        self.n_constraints = self.n_class * self.n_y_class * 2
        self.dim_condition = self.n_y_class * (self.n_class + 1)
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)
        element_K_A = self.sensitive_classes + [None]
        for i_a, a_0 in enumerate(self.sensitive_classes):
            for i_y, y_0 in enumerate(self.y_classes):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y_1 in enumerate(self.y_classes):
                        for j_a, a_1 in enumerate(element_K_A):
                            i = i_a * (2 * self.n_y_class) + i_y * 2 + i_s
                            j = j_y + self.n_y_class * j_a
                            self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                idx_true = (y == v) * (sensitive == u)
                expected_values_list.append(out[idx_true].mean())
        # sensitive is star
        for v in self.y_classes:
            idx_true = y == v
            expected_values_list.append(out[idx_true].mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y):
        # Bug B: -> return super(EqualOpportunityLoss, self).forward(X, out, sensitive, y=y)
        # Bug B Fix
        return super(AverageTreatmentEffectLoss, self).forward(X, out, sensitive, y=y)
