
import torch
from torch import nn
import torch.nn.functional as F


class _AdaptedFFN(nn.Module):
    def __init__(
        self,
        based_dim,
        ref_dim,
        num_layers,
    ):
        super().__init__()
        self.mix_layers = nn.ModuleList(
            [nn.Linear(based_dim + ref_dim, based_dim)] +
            [nn.Linear(based_dim, based_dim) for _ in range(num_layers - 1)]
        )
        self.value_linear = nn.Linear(ref_dim, based_dim)

    def forward(
        self,
        based_embeds,
        ref_embeds,
    ):
        hidden = torch.cat((based_embeds, ref_embeds), dim=-1)
        for idx, layer in enumerate(self.mix_layers):
            hidden = layer(hidden)
        weights = torch.sigmoid(hidden)
        return based_embeds + weights * self.value_linear(ref_embeds)


class FusionBase(nn.Module):
    def forward(
        self,
        text_embeds,
        audio_embeds
    ):
        raise NotImplementedError


class WeightedSumFusion(FusionBase):
    def __init__(
        self,
        weight_init_type: str = 'zero_audio',
        audio_dim=1280,
        llm_dim=2048,
    ):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        if weight_init_type == 'balance':
            self.weights = nn.Parameter(torch.tensor([1., 1.]), requires_grad=True)
        elif weight_init_type == 'zero_audio':
            self.weights = nn.Parameter(torch.tensor([-2., 2.]), requires_grad=True)
    
    def forward(
        self,
        text_embeds,  # [b, s, 3072]
        audio_embeds  # [b, s, 512]
    ):
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(2, 1, 1, 1)

        inputs = torch.stack([self.linear(audio_embeds), text_embeds], dim=0)
        fused = (weights * inputs).sum(dim=0)
        return fused


class ReferenceMixFusion(FusionBase):
    def __init__(self, 
            num_layers=3,
            audio_dim=1280,
            llm_dim=2048,
        ):
        super().__init__()
        self.layers = nn.ModuleList(
            [_AdaptedFFN(llm_dim, audio_dim, 3) for _ in range(num_layers)]
        )

    def forward(
        self,
        text_embeds,  # [b, s, 3072]
        audio_embeds  # [b, s, 1289]
    ):
        hidden = text_embeds
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, audio_embeds)
        return hidden


class SimpleSumFusion(FusionBase):
    def __init__(
        self,
        weight_init_type: str = 'zero_audio',
        audio_dim=1280,
        llm_dim=2048,
    ):
        super().__init__()
        self.in_linear = nn.Linear(audio_dim, llm_dim)
        self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)
    
    def forward(
        self,
        text_embeds,  # [b, s, 3072]
        audio_embeds  # [b, s, 512]
    ):
        return text_embeds + F.relu(self.alpha) * self.in_linear(audio_embeds)


class ExtractBase(nn.Module):
    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        raise NotImplementedError


class LinearLastExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        llm_dim=2048,
        **kwargs,
    ):
        super().__init__()
        self.linear = nn.Linear(llm_dim, k * l)
        self.k = k
        self.l = l

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        last_hidden_state = outputs.last_hidden_state.float()
        B, T, _ = last_hidden_state.shape
        flatten = self.linear(last_hidden_state)
        taste_logits = torch.reshape(flatten, (B, T, self.l, self.k))
        training_info = {}
        return taste_logits, training_info


class LinearAllConcatExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        llm_dim=2048,
        llm_num_hidden_layers=None,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(3072 * 29, k * l)
        self.k = k
        self.l = l

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        hidden = torch.cat(outputs.hidden_states, dim=-1)
        B, T, _ = hidden.shape
        flatten = self.linear(hidden)
        taste_logits = torch.reshape(flatten, (B, T, self.l, self.k))
        training_info = {}
        return taste_logits, training_info

class WeightedLayerExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        d=256,
        llm_dim=2048,
        llm_num_hidden_layers=None,
    ):
        super().__init__()
        self.num_layers = llm_num_hidden_layers + 1
        self.weights = nn.Parameter(torch.tensor([1.] * self.num_layers), requires_grad=True)
        self.linear = nn.Linear(llm_dim, k * l)
        self.k = k
        self.l = l

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(self.num_layers, 1, 1, 1)

        hidden = (weights * torch.stack(outputs.hidden_states, dim=0)).sum(dim=0)
        B, T, _ = hidden.shape
        flatten = self.linear(hidden)
        taste_logits = torch.reshape(flatten, (B, T, self.l, self.k))
        training_info = {}
        return taste_logits, training_info


class ReferenceMixExtract(ExtractBase):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [_AdaptedFFN(512, 3072, 3) for _ in range(num_layers)]
        )

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        ref_hidden = outputs.last_hidden_state
        hidden = audio_embeds
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, ref_hidden)

        taste_logits = hidden
        training_info = {}
        return taste_logits, training_info
    

class ContinueWeightedLayerExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        d=256,
        llm_dim=2048,
        llm_num_hidden_layers=None,
    ):
        super().__init__()
        self.k = k
        self.num_layers = llm_num_hidden_layers + 1
        self.weights = nn.Parameter(torch.tensor([1.] * self.num_layers), requires_grad=True)
        self.linear = nn.Linear(llm_dim, d)

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(self.num_layers, 1, 1, 1)

        hidden = (weights * torch.stack(outputs.hidden_states, dim=0)).sum(dim=0)
        hidden = self.linear(hidden)

        predicted_llm_indices, _ = vq_module.get_indices_from_code(hidden)
        taste_logits = F.one_hot(predicted_llm_indices, num_classes=self.k) * 1000.0

        training_info = {}
        return taste_logits, training_info


class ContinueLatentWeightedLayerExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        d=256,
        llm_dim=2048,
        llm_num_hidden_layers=None,

        fc_mu_requires_bias=True,
        b_logvar_is_linear=False,
        use_additional_mlp_and_residual=False,
        conduct_reparameterization=True,
    ):
        super().__init__()
        self.k = k
        self.num_layers = llm_num_hidden_layers + 1
        self.weights = nn.Parameter(torch.tensor([1.] * self.num_layers), requires_grad=True)

        self.fc_mu = nn.Linear(llm_dim, d, bias=fc_mu_requires_bias)
        if b_logvar_is_linear:
            self.b_logvar = nn.Linear(llm_dim, d, bias=False)
        else:
            self.b_logvar = nn.Parameter(torch.zeros(d))
        self.b_logvar_is_linear = b_logvar_is_linear
        self.fc_mu_requires_bias = fc_mu_requires_bias
        self.use_additional_mlp_and_residual = use_additional_mlp_and_residual
        self.conduct_reparameterization = conduct_reparameterization
        if self.use_additional_mlp_and_residual:
            self.mlp = nn.Sequential(
                [
                    nn.Linear(d)
                ]
            )

    def reparameterize(self, mu, sigma):
        orig_value = mu + sigma
        epsilon = torch.randn_like(sigma)
        _sampled_value = mu + sigma * epsilon
        new_value = orig_value + (_sampled_value - orig_value).detach()
        return new_value

    def _layer_weighted_sum(self, hidden_states):
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(self.num_layers, 1, 1, 1)

        hidden = (weights * torch.stack(hidden_states, dim=0)).sum(dim=0)
        return hidden

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        hidden = self._layer_weighted_sum(outputs.hidden_states)

        # compute the mean
        mu = self.fc_mu(hidden)
        # get the logvar
        if self.b_logvar_is_linear:
            logvar = self.b_logvar(hidden)
            sigma = torch.exp(0.5 * logvar)
        else:
            logvar = self.b_logvar
            sigma = torch.exp(0.5 * logvar).view(1, 1, -1).expand_as(mu)
        # reparameterize to get the sampled latent z
        if self.training and self.conduct_reparameterization:
            z = self.reparameterize(mu, sigma)
        else:
            z = mu + sigma
        # y_pred = self.mlp(z) + z # currently we don't have additional mlp for residual 

        predicted_llm_indices, _ = vq_module.get_indices_from_code(z)
        taste_logits = F.one_hot(predicted_llm_indices, num_classes=self.k) * 1000.0

        training_info = {
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
        return taste_logits, training_info


class ContinueLatentLinearLastExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        d=256,
        llm_dim=2048,
        llm_num_hidden_layers=None,

        fc_mu_requires_bias=True,
        b_logvar_is_linear=False,
        use_additional_mlp_and_residual=False,
        conduct_reparameterization=True,
    ):
        super().__init__()
        self.k = k

        self.fc_mu = nn.Linear(llm_dim, d, bias=fc_mu_requires_bias)
        if b_logvar_is_linear:
            self.b_logvar = nn.Linear(llm_dim, d, bias=False)
        else:
            self.b_logvar = nn.Parameter(torch.zeros(d))
        self.b_logvar_is_linear = b_logvar_is_linear
        self.fc_mu_requires_bias = fc_mu_requires_bias
        self.use_additional_mlp_and_residual = use_additional_mlp_and_residual
        self.conduct_reparameterization = conduct_reparameterization
        if self.use_additional_mlp_and_residual:
            self.mlp = nn.Sequential(
                [
                    nn.Linear(d)
                ]
            )

    def reparameterize(self, mu, sigma):
        orig_value = mu + sigma
        epsilon = torch.randn_like(sigma)
        _sampled_value = mu + sigma * epsilon
        new_value = orig_value + (_sampled_value - orig_value).detach()
        return new_value

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):
        hidden = outputs.last_hidden_state.float()

        # compute the mean
        mu = self.fc_mu(hidden)
        # get the logvar
        if self.b_logvar_is_linear:
            logvar = self.b_logvar(hidden)
            sigma = torch.exp(0.5 * logvar)
        else:
            logvar = self.b_logvar
            sigma = torch.exp(0.5 * logvar).view(1, 1, -1).expand_as(mu)
        # reparameterize to get the sampled latent z
        if self.training and self.conduct_reparameterization:
            z = self.reparameterize(mu, sigma)
        else:
            z = mu + sigma
        # y_pred = self.mlp(z) + z # currently we don't have additional mlp for residual 

        vq_module.eval()
        predicted_llm_indices = vq_module.get_indices_from_code(z)
        taste_logits = F.one_hot(predicted_llm_indices, num_classes=self.k) * 1000.0

        training_info = {
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
        return taste_logits, training_info


class MultiLinearLastExtract(ExtractBase):
    def __init__(
        self,
        k=512,
        l=4,
        d=256,
        llm_dim=2048,
        llm_num_hidden_layers=None,

        fc_mu_requires_bias=True,
        conduct_reparameterization=True,
    ):
        super().__init__()
        self.k = k
        self.l = l
        self.d = d

        self.linear = nn.Linear(llm_dim, k * l)
        self.b_logvar = nn.Parameter(torch.zeros(d * l))

        self.fc_mu_requires_bias = fc_mu_requires_bias
        self.conduct_reparameterization = conduct_reparameterization

    def reparameterize(self, mu, sigma):
        orig_value = mu + sigma
        epsilon = torch.randn_like(sigma)
        _sampled_value = mu + sigma * epsilon
        new_value = orig_value + (_sampled_value - orig_value).detach()
        return new_value

    def forward(
        self,
        outputs,
        vq_module=None,
        audio_embeds=None,
    ):

        last_hidden_state = outputs.last_hidden_state.float()
        B, T, _ = last_hidden_state.shape
        flatten = self.linear(last_hidden_state)
        taste_logits = torch.reshape(flatten, (B, T, self.l, self.k))

        codes = vq_module.get_distributed_codes(taste_logits) # [b, t, l, d]

        training_info = {
            'agg_code': codes.sum(dim=-2)
        }
        return taste_logits, training_info


BRIDGE_FUSION_CLASSES = {
    'weighted_sum': WeightedSumFusion,
    'reference_mix': ReferenceMixFusion,
    'simple_sum': SimpleSumFusion,
}
BRIDGE_EXTRACT_CLASSES = {
    'linear_last': LinearLastExtract,
    'linear_all_concat': LinearAllConcatExtract,
    'reference_mix': ReferenceMixExtract,
    'weighted_layer': WeightedLayerExtract,

    'continue_weighted_layer': ContinueWeightedLayerExtract,
    'continue_latent_weighted_layer': ContinueLatentWeightedLayerExtract,
    'continue_latent_linear_last': ContinueLatentLinearLastExtract,

    'multi_linear_last': MultiLinearLastExtract,
}
