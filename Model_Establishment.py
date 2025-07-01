import torch
import torch.nn as nn
from Networks import Generator, MLP_classifier, CD_encoder, DIFM


class CD_Net(nn.Module):
    def __init__(self, input_dim, encoder_param):
        super(CD_Net, self).__init__()

        self.mlp_classifier = MLP_classifier(input_dim)
        self.cd_encoder_a = CD_encoder(32, 16)
        self.cd_encoder_b = CD_encoder(32, 16)
        self.difm = DIFM(16)

        self.parameters_list_classifier = encoder_param + list(self.mlp_classifier.parameters()) + list(self.cd_encoder_a.parameters()) + list(self.cd_encoder_b.parameters()) + list(self.difm.parameters())

        self.criterion = torch.nn.CrossEntropyLoss()

    def update(self, enc_content_a, enc_content_b, x_a, x_b, x_ab, x_ba, y):

        pred, train_pred = self.forward(enc_content_a, enc_content_b, x_a, x_b, x_ab, x_ba)

        loss = self.criterion(pred, y)

        return loss, train_pred

    def forward(self, enc_content_a, enc_content_b, x_a, x_b, x_ab, x_ba):
        c_a = self.cd_encoder_a(enc_content_a.forward_3(x_a))
        c_b = self.cd_encoder_b(enc_content_b.forward_3(x_b))

        c_ba = self.cd_encoder_a(enc_content_a.forward_3(x_ba))
        c_ab = self.cd_encoder_b(enc_content_b.forward_3(x_ab))

        c = self.difm(c_a, c_ba, c_b, c_ab)

        pred = self.mlp_classifier(c)

        train_pred = torch.argmax(pred, dim=1)

        return pred, train_pred


class Trans_Net(nn.Module):
    def __init__(self, a_channel, b_channel):
        super(Trans_Net, self).__init__()

        self.gen_a = Generator(a_channel)
        self.gen_b = Generator(b_channel)

        self.parameters_list = nn.ParameterList([*self.gen_a.parameters(), *self.gen_b.parameters()])
        self.con_encoder_params = list(self.gen_a.enc_content.parameters()) + list(self.gen_b.enc_content.parameters())

    def recon_criterion(self, input, target):
        return torch.square(input-target).mean()

    def recon_criterion_mask(self, input, target, mask):

        # Pu = (mask == 0.0).float()
        Pu = (1.0 - mask).float()

        unchanged_loss = torch.mean(torch.square(input - target) * Pu)

        return unchanged_loss

    def update(self, x_a, x_b, mask):

        # encode (within domain)
        c_a, s_a = self.gen_a.encode(x_a)
        c_b, s_b = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        x_a_recon = self.gen_a.decode(c_a, s_a)
        x_b_recon = self.gen_b.decode(c_b, s_b)

        # encode (reconstruction)
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # decode (within domain) not use
        x_aba = self.gen_a.decode(c_a_recon, s_a_recon)
        x_bab = self.gen_b.decode(c_b_recon, s_b_recon)

        # self-reconstruction loss
        loss_recon = self.recon_criterion(x_a, x_a_recon) + self.recon_criterion(x_b, x_b_recon)

        # image translation loss
        loss_trans = (self.recon_criterion(c_a, c_a_recon) + self.recon_criterion(s_b, s_b_recon)
                      + self.recon_criterion(c_b, c_b_recon) + self.recon_criterion(s_a, s_a_recon))

        # content code alignment loss
        loss_align = (self.recon_criterion_mask(c_a_recon, c_b, mask)
                      + self.recon_criterion_mask(c_b_recon, c_a, mask))

        # modality consistency loss
        loss_align_img = (self.recon_criterion_mask(x_a, x_ba, mask)
                      + self.recon_criterion_mask(x_b, x_ab, mask))

        # Total loss
        weight_recon = 10
        weight_trans = 1
        weight_align = 1
        weight_align_img = 1

        total_loss = (weight_recon * loss_recon +
                      weight_trans * loss_trans +
                      weight_align * loss_align +
                      weight_align_img * loss_align_img)

        return total_loss

    def forward(self, x_a, x_b):

        # encode (within domain)
        c_a, s_a = self.gen_a.encode(x_a)
        c_b, s_b = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        x_a_recon = self.gen_a.decode(c_a, s_a)
        x_b_recon = self.gen_b.decode(c_b, s_b)

        # encode (reconstruction)
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        return {'x_ba': x_ba, 'x_ab': x_ab, 'x_a_recon': x_a_recon, 'x_b_recon': x_b_recon,
                'c_a': c_a, 'c_b': c_b, 'c_a_recon': c_a_recon, 'c_b_recon': c_b_recon
                }
