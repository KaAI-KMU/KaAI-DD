import torch
import torch.nn as nn
from torch.autograd import Variable



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[step]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class Decoder(nn.Module):
    def  __init__(self, num_step, num_channel):
        super(Decoder, self).__init__()
        self._all_layers = []
        self.num_step = num_step
        self.num_channel = num_channel
        for i in range(self.num_step):
            name = 'conv{}'.format(i)
            conv = nn.Conv2d(self.num_channel, 3, 1, stride=1, padding=0)
            setattr(self, name, conv)
            self._all_layers.append(conv)

    def forward(self, inputs):
        # inputs가 [배치 크기, 채널, 높이, 너비] 형태일 경우, [배치 크기, 1, 채널, 높이, 너비]로 변환
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)  # 시간 차원 추가
        
        batch_size, num_steps, _, height, width = inputs.size()
        outputs = []

        for i in range(num_steps):
            x = inputs[:, i, :, :, :]  # [배치 크기, 채널, 높이, 너비]
            y = self._all_layers[i % len(self._all_layers)](x)
            outputs.append(y.unsqueeze(1))  # 시간 차원 추가

        output_tensor = torch.cat(outputs, dim=1)  # [배치 크기, 시간 단계, 채널, 높이, 너비]
        # 시간 차원을 배치 차원에 합치기 위해 view 사용
        output_tensor = output_tensor.view(batch_size * num_steps, 3, height, width)
        return output_tensor

class Encoder(nn.Module):
    def __init__(self, hidden_channels, sample_size, sample_duration):
        super(Encoder, self).__init__()
        self.convlstm = ConvLSTM(input_channels=3, hidden_channels=hidden_channels, kernel_size=3, step=sample_duration,
                        effective_step=[sample_duration-1])
################## W/o output decoder
        self.conv2 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        # )
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        # )
        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        # )
        # self.conv_block4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        # )
        # self.conv_block5 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
            
        # )
################## With output decoder
        # self.decoder = Decoder(16, 32)
    def forward(self, x):
        b,t,c,h,w = x.size()
        x = x.permute(1,0,2,3,4)
        output_convlstm, _ = self.convlstm(x)
        
        # x = self.decoder(output_convlstm[0])
        x = self.conv2(output_convlstm[0])
        # x = self.conv_block(x)
        # x = self.conv_block2(x)
        # x = self.conv_block3(x)
        # x = self.conv_block4(x)
        # x = self.conv_block5(x)
        # #flatten the output
        # x = x.view(b,-1)
        return x
    
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        # FC0: 3072 -> 2048
        
        self.classifier_fc = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        x = self.classifier_fc(x)
        return x
		

def test():
#if __name__ == '__main__':
    # gradient check

    convlstm = ConvLSTM(input_channels=48, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[2,4]).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 48, 64, 64)).cuda()
    target = Variable(torch.randn(1, 32, 64, 64)).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()

    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)


def test_convlstm():
    """Constructs a convlstm model.
    """
    model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=[112,112], sample_duration=4).cuda()
    input = Variable(torch.randn(20, 3, 4, 112, 112)).cuda()

    output = model(input)
    print(output.size())

def encoder(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Encoder(**kwargs)
    return model

#if __name__ == '__main__':
#    test_convlstm()