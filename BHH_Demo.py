import torch
import matplotlib.pyplot as plt

##############################################################################################################
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 16,
        "figure.dpi": 100,
    }
)

# Most of this notebook can be run on CPU in a reasonable amount of time.
# The example training at the end cannot be.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
###########################################################################################################3
# 期望的时域波形持续时间
waveform_duration = 8
# 我们今天将使用的所有数据的采样率
sample_rate = 2048

# 定义最小频率、最大频率和参考频率
f_min = 20
f_max = 1024
f_ref = 20

nyquist = sample_rate / 2
num_samples = int(waveform_duration * sample_rate)
num_freqs = num_samples // 2 + 1

# 创建一个频率值数组，用于生成波形
# 目前，仅实现了频域近似方法
frequencies = torch.linspace(0, nyquist, num_freqs).to(device)
freq_mask = (frequencies >= f_min) * (frequencies < f_max).to(device)
#################################################################################################################
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
from torch.distributions import Uniform

# 在 CPU 上，保持波形数量大约为 100；在 GPU 上可以使用更大数量，
# 但需要受限于显存大小。
num_waveforms = 500

# 创建一个参数分布的字典
# 注意：这些分布并不是天体物理学上真实有意义的分布
param_dict = {
    "chirp_mass": PowerLaw(10, 100, -2.35),   # 啁啾质量，使用幂律分布
    "mass_ratio": Uniform(0.125, 0.999),      # 质量比，使用均匀分布
    "chi1": Uniform(-0.999, 0.999),           # 自旋参数 chi1，均匀分布
    "chi2": Uniform(-0.999, 0.999),           # 自旋参数 chi2，均匀分布
    "distance": PowerLaw(100, 1000, 2),       # 距离，使用幂律分布
    "phic": DeltaFunction(0),                 # 相位常数，固定为 0
    "inclination": Sine(),                    # 倾角，使用正弦分布
}

# 然后从这些分布中采样参数
params = {
    k: v.sample((num_waveforms,)).to(device) for k, v in param_dict.items()
}
#########################################################################################################
from ml4gw.waveforms import IMRPhenomD

approximant = IMRPhenomD().to(device)

# 调用近似模型时，传入频率数组、参考频率和波形参数，
# 将返回交叉极化 (cross) 和加号极化 (plus) 两个波形分量
hc_f, hp_f = approximant(f=frequencies[freq_mask], f_ref=f_ref, **params)
print(hc_f.shape, hp_f.shape)
#########################################################################################################

# 注意：为了绘图，我需要把数据移到 CPU 上
# 在实际的训练环境中，不会频繁地在不同设备之间移动数据
plt.plot(frequencies[freq_mask].cpu(), torch.abs(hp_f[0]).cpu())
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strain")
plt.show()
#########################################################################################################

from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components

waveform_generator = TimeDomainCBCWaveformGenerator(
    approximant=approximant,
    sample_rate=sample_rate,
    f_min=f_min,
    duration=waveform_duration,
    right_pad=0.5,
    f_ref=f_ref,
).to(device)

params["mass_1"], params["mass_2"] = chirp_mass_and_mass_ratio_to_components(
    params["chirp_mass"], params["mass_ratio"]
)

params["s1z"], params["s2z"] = params["chi1"], params["chi2"]

hc, hp = waveform_generator(**params)
print(hc.shape, hp.shape)
#########################################################################################################

times = torch.arange(0, waveform_duration, 1 / sample_rate)
plt.plot(times, hp[0].cpu())
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.show()
#########################################################################################################

from ml4gw.gw import get_ifo_geometry, compute_observed_strain

# 定义天空位置与偏振角的概率分布
dec = Cosine()                     # 赤纬分布（余弦分布）
psi = Uniform(0, torch.pi)         # 偏振角分布（0 到 π 的均匀分布）
phi = Uniform(-torch.pi, torch.pi) # 赤经分布（-π 到 π 的均匀分布）

# 干涉仪几何结构，ml4gw 中也包含 V1 和 K1 的配置
ifos = ["H1", "L1"]
tensors, vertices = get_ifo_geometry(*ifos)

# 将探测器几何信息、极化态和天空参数传入，
# 计算得到探测器实际观测到的引力波应变信号
waveforms = compute_observed_strain(
    dec=dec.sample((num_waveforms,)).to(device),
    psi=psi.sample((num_waveforms,)).to(device),
    phi=phi.sample((num_waveforms,)).to(device),
    detector_tensors=tensors.to(device),
    detector_vertices=vertices.to(device),
    sample_rate=sample_rate,
    cross=hc,
    plus=hp,
)
print(waveforms.shape)
#########################################################################################################

plt.plot(times, waveforms[0, 0].cpu(), label="H1", alpha=0.5)
plt.plot(times, waveforms[0, 1].cpu(), label="L1", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.legend()
plt.show()
#########################################################################################################

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path

# 指定保存本 Notebook 生成的所有数据产品的目录
data_dir = Path("./data")

# 指定下载数据的目录
background_dir = data_dir / "background_data"
background_dir.mkdir(parents=True, exist_ok=True)

# 这些是数据片段的 GPS 起始和结束时间
# 选择这些时间没有特别原因，只是因为它们包含可直接用于分析的数据
segments = [
    (1240579783, 1240587612),
    (1240594562, 1240606748),
    (1240624412, 1240644412),
    (1240644412, 1240654372),
    (1240658942, 1240668052),
]

for (start, end) in segments:
    # 从 GWOSC 下载数据，这一步可能需要几分钟
    duration = end - start
    fname = background_dir / f"background-{start}-{duration}.hdf5"
    if fname.exists():
        continue

    ts_dict = TimeSeriesDict()
    for ifo in ifos:
        ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
    ts_dict = ts_dict.resample(sample_rate)   # 重新采样到设定的采样率
    ts_dict.write(fname, format="hdf5")       # 保存为 hdf5 格式文件
#########################################################################################################

from ml4gw.transforms import SpectralDensity
import h5py

fftlength = 2
spectral_density = SpectralDensity(
    sample_rate=sample_rate,
    fftlength=fftlength,
    overlap=None,
    average="median",
).to(device)

# 这是之前下载的 O3 观测期的 H1 和 L1 数据
# 后面会介绍用于数据加载的工具
background_file = background_dir / "background-1240579783-7829.hdf5"
with h5py.File(background_file, "r") as f:
    background = [torch.Tensor(f[ifo][:]) for ifo in ifos]
    background = torch.stack(background).to(device)

# 注意：这里转换为 double 类型
psd = spectral_density(background.double())
print(psd.shape)
#########################################################################################################

freqs = torch.linspace(0, nyquist, psd.shape[-1])
plt.plot(freqs, psd.cpu()[0], label="H1")
plt.plot(freqs, psd.cpu()[1], label="L1")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (1/Hz)")
plt.legend()
plt.show()
#########################################################################################################
from ml4gw.gw import compute_ifo_snr, compute_network_snr

# 注意：这里需要对 PSD 进行插值
if psd.shape[-1] != num_freqs:
    # 为保持维度一致性，先添加虚拟维度
    while psd.ndim < 3:
        psd = psd[None]
    psd = torch.nn.functional.interpolate(
        psd, size=(num_freqs,), mode="linear"
    )

# 我们可以计算单个干涉仪的 SNR 和网络 SNR
# SNR 计算从之前设定的最小频率开始，到最大频率结束
# TODO：可能没必要写成多个函数
h1_snr = compute_ifo_snr(
    responses=waveforms[:, 0],
    psd=psd[:, 0],
    sample_rate=sample_rate,
    highpass=f_min,
)
l1_snr = compute_ifo_snr(
    responses=waveforms[:, 1],
    psd=psd[:, 1],
    sample_rate=sample_rate,
    highpass=f_min,
)
network_snr = compute_network_snr(
    responses=waveforms, psd=psd, sample_rate=sample_rate, highpass=f_min
)
#########################################################################################################
plt.hist(h1_snr.cpu(), bins=25, alpha=0.5, label="H1")
plt.hist(l1_snr.cpu(), bins=25, alpha=0.5, label="L1")
plt.hist(network_snr.cpu(), bins=25, alpha=0.5, label="Network")
plt.xlabel("SNR")
plt.ylabel("Count")
plt.xlim(0, 100)
plt.legend()
plt.show()
#########################################################################################################
from ml4gw.gw import reweight_snrs

# 采样目标 SNR，服从幂律分布，范围 12 到 100
target_snrs = PowerLaw(12, 100, -3).sample((num_waveforms,)).to(device)

# 每个波形都会按照目标 SNR 与当前 SNR 的比值进行缩放
waveforms = reweight_snrs(
    responses=waveforms,
    target_snrs=target_snrs,
    psd=psd,
    sample_rate=sample_rate,
    highpass=f_min,
)

# 重新计算缩放后的网络 SNR
network_snr = compute_network_snr(
    responses=waveforms, psd=psd, sample_rate=sample_rate, highpass=f_min
)

# 绘制网络 SNR 的直方图分布
plt.hist(network_snr.cpu(), bins=25, alpha=0.5, label="Network")
plt.xlabel("SNR")
plt.ylabel("Count")
plt.xlim(0, 100)
plt.legend()
plt.show()
#########################################################################################################
from ml4gw.dataloading import Hdf5TimeSeriesDataset

# 定义一些参数，便于后续使用，并确定采样窗口的大小
# 我们将对白化窗口的后半部分进行处理，使用前半部分计算得到的 PSD，
# 因此需要采集足够的数据来完成这一过程

# 用于估计 PSD 的数据长度
psd_length = 16
psd_size = int(psd_length * sample_rate)

# 滤波器长度。白化后会从两端各裁掉 fduration / 2 的数据
fduration = 2

# 输入到神经网络的数据窗口长度
kernel_length = 1.5
kernel_size = int(1.5 * sample_rate)

# 总的数据采样长度
window_length = psd_length + fduration + kernel_length

# 读取背景数据文件
fnames = list(background_dir.iterdir())
dataloader = Hdf5TimeSeriesDataset(
    fnames=fnames,
    channels=ifos,
    kernel_size=int(window_length * sample_rate),
    batch_size=2 * num_waveforms,  # 获取的背景样本数量为波形数的两倍
    batches_per_epoch=1,           # 这里只设置 1 批次，演示用途
    coincident=False,              # 不强制要求不同干涉仪采样时间重合
)

# 从数据加载器中获取一个批次的样本
background_samples = [x for x in dataloader][0].to(device)
print(background_samples.shape)
#########################################################################################################
from ml4gw.transforms import Whiten

whiten = Whiten(
    fduration=fduration, sample_rate=sample_rate, highpass=f_min
).to(device)

# 使用每个样本前 psd_length 秒的数据生成 PSD，
# 调用我们之前定义的 SpectralDensity 模块
psd = spectral_density(background_samples[..., :psd_size].double())
print(f"PSD shape: {psd.shape}")

# 将 psd_length 之后的部分作为输入数据（kernel）
kernel = background_samples[..., psd_size:]

# 使用生成的 PSD 对输入数据进行白化
whitened_kernel = whiten(kernel, psd)
print(f"Kernel shape: {kernel.shape}")
print(f"Whitened kernel shape: {whitened_kernel.shape}")
#########################################################################################################
times = torch.arange(0, kernel_length + fduration, 1 / sample_rate)
plt.plot(times, kernel[0, 0].cpu())
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.show()

times = torch.arange(0, kernel_length, 1 / sample_rate)
plt.plot(times, whitened_kernel[0, 0].cpu())
plt.xlabel("Time (s)")
plt.ylabel("Whitened strain")
plt.show()
#########################################################################################################
pad = int(fduration / 2 * sample_rate)
injected = kernel.detach().clone()
# Inject waveforms into every other background sample
injected[::2, :, pad:-pad] += waveforms[..., -kernel_size:]
# And whiten with the same PSDs as before
whitened_injected = whiten(injected, psd)
#########################################################################################################
# Factor of 2 because we injected every other sample
idx = 2 * torch.argmax(network_snr)

times = torch.arange(0, kernel_length + fduration, 1 / sample_rate)
plt.plot(times, injected[idx, 0].cpu())
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.show()

times = torch.arange(0, kernel_length, 1 / sample_rate)
plt.plot(times, whitened_injected[idx, 0].cpu())
plt.xlabel("Time (s)")
plt.ylabel("Whitened strain")
plt.show()
#########################################################################################################
y = torch.zeros(len(injected))
y[::2] = 1
with h5py.File(data_dir / "validation_dataset.hdf5", "w") as f:
    f.create_dataset("X", data=whitened_injected.cpu())
    f.create_dataset("y", data=y)
#########################################################################################################
from ml4gw.transforms import (
    MultiResolutionSpectrogram,
    QScan,
    SingleQTransform,
)

mrs = MultiResolutionSpectrogram(
    kernel_length=kernel_length,
    sample_rate=sample_rate,
    n_fft=[
        64,
        128,
        256,
    ],  # Specififying just one value will create a single-resolution spectrogram
).to(device)

# The Q-transform can be accessed either through the QScan,
# which will look over a range of q values, or through the
# SingleQTransform, which uses a given q value
qscan = QScan(
    duration=kernel_length,
    sample_rate=sample_rate,
    spectrogram_shape=[512, 512],
    qrange=[4, 128],
).to(device)
sqt = SingleQTransform(
    duration=kernel_length,
    sample_rate=sample_rate,
    spectrogram_shape=[512, 512],
    q=12,
).to(device)
#########################################################################################################
specgram = mrs(whitened_injected)
plt.imshow(specgram[idx, 0].cpu(), aspect="auto", origin="lower")
plt.show()
#########################################################################################################
specgram = qscan(whitened_injected)
plt.imshow(specgram[idx, 0].cpu(), aspect="auto", origin="lower")
plt.show()
#########################################################################################################
specgram = sqt(whitened_injected)
plt.imshow(specgram[idx, 0].cpu(), aspect="auto", origin="lower")
plt.show()
#########################################################################################################
from ml4gw.nn.resnet import ResNet1D

architecture = ResNet1D(
    in_channels=2,   # 输入通道为 2，对应 H1 和 L1
    layers=[2, 2],   # 保持较小规模，这里相当于 ResNet10
    classes=1,       # 输出为单个标量
    kernel_size=3,   # 卷积核大小（不要与数据窗口大小混淆）
).to(device)

# 我们可以例如传入验证集中的第一个样本进行测试
with torch.no_grad():
    print(architecture(whitened_injected[0][None]))
#########################################################################################################
import gc, torch

gc.collect()
torch.cuda.empty_cache()

# 安全调用：不存在就跳过
try:
    torch.cuda.reset_peak_memory_stats()
except Exception as e:
    print(f"[Warn] skip reset_peak_memory_stats: {e}")
#########################################################################################################
from ml4gw import augmentations, distributions, gw, transforms, waveforms
from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
from ml4gw.utils.slicing import sample_kernels
import torch
from lightning import pytorch as pl
import torchmetrics
from torchmetrics.classification import BinaryAUROC

from typing import Callable, Dict, List


class Ml4gwDetectionModel(pl.LightningModule):
    """
    模型：包含在 GPU 上实时生成波形并执行预处理增强的方法；
    同时从磁盘分块加载训练用的背景数据，再从这些分块中采样批次进行训练。
    """

    def __init__(
            self,
            architecture: torch.nn.Module,
            metric: torchmetrics.Metric,
            ifos: List[str] = ["H1", "L1"],
            kernel_length: float = 1.5,
            # PSD/白化参数
            fduration: float = 2,
            psd_length: float = 16,
            sample_rate: float = 2048,
            fftlength: float = 2,
            highpass: float = 32,
            # 数据加载参数
            chunk_length: float = 128,  # 我们稍后会介绍“chunk（分块）”的概念
            reads_per_chunk: int = 40,
            learning_rate: float = 0.005,
            batch_size: int = 256,
            # 波形生成参数
            waveform_prob: float = 0.5,
            approximant: Callable = waveforms.cbc.IMRPhenomD,
            param_dict: Dict[str, torch.distributions.Distribution] = param_dict,
            waveform_duration: float = 8,
            f_min: float = 20,
            f_max: float = None,
            f_ref: float = 20,
            # 增广参数
            inversion_prob: float = 0.5,
            reversal_prob: float = 0.5,
            min_snr: float = 12,
            max_snr: float = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["architecture", "metric", "approximant"]
        )
        self.nn = architecture
        self.metric = metric

        self.inverter = augmentations.SignalInverter(prob=inversion_prob)   # 信号取反增广
        self.reverser = augmentations.SignalReverser(prob=reversal_prob)    # 时间反转增广

        # 使用 torch Module 定义的实时变换
        self.spectral_density = transforms.SpectralDensity(
            sample_rate, fftlength, average="median", fast=False
        )
        self.whitener = transforms.Whiten(
            fduration, sample_rate, highpass=highpass
        )

        # 获取将要投影到的干涉仪的几何信息
        detector_tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("detector_tensors", detector_tensors)
        self.register_buffer("detector_vertices", vertices)

        # 定义天空参数分布
        self.param_dict = param_dict
        self.dec = distributions.Cosine()                    # 赤纬余弦分布
        self.psi = torch.distributions.Uniform(0, torch.pi)  # 偏振角均匀分布
        self.phi = torch.distributions.Uniform(
            -torch.pi, torch.pi
        )  # 探测器与源的相对赤经（RA）

        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant=approximant(),
            sample_rate=sample_rate,
            duration=waveform_duration,
            f_min=f_min,
            f_ref=f_ref,
            right_pad=0.5,
        ).to(self.device)

        # 与其直接采样距离，这里改为采样目标 SNR
        # 这样可以确保训练的信号更“可探测”
        # 分布大致与自然采样到的 SNR 分布相似
        self.snr = distributions.PowerLaw(min_snr, max_snr, -3)

        # 预先用“样本数”单位定义若干属性
        # 注意：这里的 window_size 与前文的用法不同
        self.kernel_size = int(kernel_length * sample_rate)
        self.window_size = self.kernel_size + int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)

    def forward(self, X):
        return self.nn(X)

    def training_step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        self.metric.update(y_hat, y)
        self.log("valid_auroc", self.metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        parameters = self.nn.parameters()
        optimizer = torch.optim.AdamW(parameters, self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            pct_start=0.1,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = dict(scheduler=scheduler, interval="step")
        return dict(optimizer=optimizer, lr_scheduler=scheduler_config)

    def configure_callbacks(self):
        chkpt = pl.callbacks.ModelCheckpoint(monitor="valid_auroc", mode="max")
        return [chkpt]

    def generate_waveforms(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        rvs = torch.rand(size=(batch_size,))
        mask = rvs < self.hparams.waveform_prob
        num_injections = mask.sum().item()

        params = {
            k: v.sample((num_injections,)).to(device)
            for k, v in self.param_dict.items()
        }

        # 将自旋与质量参数转换为生成器所需的名字/格式
        params["s1z"], params["s2z"] = (
            params["chi1"], params["chi2"]
        )
        params["mass_1"], params["mass_2"] = waveforms.conversion.chirp_mass_and_mass_ratio_to_components(
            params["chirp_mass"], params["mass_ratio"]
        )

        hc, hp = self.waveform_generator(**params)
        return hc, hp, mask

    def project_waveforms(
            self, hc: torch.Tensor, hp: torch.Tensor
    ) -> torch.Tensor:
        # 采样天空位置参数
        N = len(hc)
        dec = self.dec.sample((N,)).to(hc)
        psi = self.psi.sample((N,)).to(hc)
        phi = self.phi.sample((N,)).to(hc)

        # 投影到干涉仪的响应（得到各台干涉仪的应变）
        return gw.compute_observed_strain(
            dec=dec,
            psi=psi,
            phi=phi,
            detector_tensors=self.detector_tensors,
            detector_vertices=self.detector_vertices,
            sample_rate=self.hparams.sample_rate,
            cross=hc,
            plus=hp,
        )

    def rescale_snrs(
            self, responses: torch.Tensor, psd: torch.Tensor
    ) -> torch.Tensor:
        # 确保 PSD 与响应的频率 bin 数一致
        num_freqs = int(responses.size(-1) // 2) + 1
        if psd.size(-1) != num_freqs:
            psd = torch.nn.functional.interpolate(
                psd, size=(num_freqs,), mode="linear"
            )
        N = len(responses)
        target_snrs = self.snr.sample((N,)).to(responses.device)
        return gw.reweight_snrs(
            responses=responses.double(),
            target_snrs=target_snrs,
            psd=psd,
            sample_rate=self.hparams.sample_rate,
            highpass=self.hparams.highpass,
        )

    def sample_waveforms(self, responses: torch.Tensor) -> torch.Tensor:
        # 从每个波形尾部截取随机视图，以便在任意位置注入
        responses = responses[:, :, -self.window_size :]

        # 进行 padding，保证至少有一半 kernel 含有信号
        pad = [0, int(self.window_size // 2)]
        responses = torch.nn.functional.pad(responses, pad)
        return sample_kernels(responses, self.window_size, coincident=True)

    @torch.no_grad()
    def augment(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 从目标 kernel 前面截取“背景”片段并计算其 PSD
        # （由于幅度尺度很小，这里使用双精度）
        background, X = torch.split(
            X, [self.psd_size, self.window_size], dim=-1
        )
        psd = self.spectral_density(background.double())

        # 最多生成 batch_size 条信号；保留一个 mask 表示哪些样本需要注入
        batch_size = X.size(0)
        hc, hp, mask = self.generate_waveforms(batch_size)
        hc, hp, mask = hc, hp, mask

        # 执行取反与时间反转增广
        X = self.inverter(X)
        X = self.reverser(X)

        # 采样天空参数并投影为各台干涉仪的响应，
        # 然后根据随机采样到的目标 SNR 进行幅度重标定
        responses = self.project_waveforms(hc, hp)
        responses = self.rescale_snrs(responses, psd[mask])

        # 随机裁剪一段波形窗口，将其叠加到背景上，然后对白化
        responses = self.sample_waveforms(responses)
        X[mask] += responses.float()
        X = self.whitener(X, psd)

        # 生成标签：在发生注入的位置标记为 1
        y = torch.zeros((batch_size, 1), device=X.device)
        y[mask] = 1
        return X, y

    def on_after_batch_transfer(self, batch, _):
        # 这是 Lightning 在批次移动到 GPU 与进入 training_step 之间调用的父方法；
        # 在这里应用我们的数据增广流程
        if self.trainer.training:
            batch = self.augment(batch)
        return batch

    def train_dataloader(self):
        # 因为整个训练集是“即时生成”的，
        # 传统意义上“一个 epoch 遍历一次训练集”的概念不再适用。
        # 我们需要自己设定每个 epoch 的批次数，这实质上决定
        # 多久在验证集上评估一次。
        samples_per_epoch = 3000
        batches_per_epoch = (
                int((samples_per_epoch - 1) // self.hparams.batch_size) + 1
        )
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1

        # Hdf5TimeSeriesDataset 从磁盘采样批次。
        # 这里我们把“批次”设得很大，把它们看作“分块（chunk）”，
        # 后续再从这些分块中采样出真正用于训练的小批次
        fnames = list(background_dir.iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.hparams.ifos,
            kernel_size=int(
                self.hparams.chunk_length * self.hparams.sample_rate
            ),
            batch_size=self.hparams.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )

        # 从磁盘加载的分块里再采样批次，喂给神经网络
        return ChunkedTimeSeriesDataset(
            dataset,
            kernel_size=self.window_size + self.psd_size,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=False,
        )

    def val_dataloader(self):
        with h5py.File(data_dir / "validation_dataset.hdf5", "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 4,
            shuffle=False,
            pin_memory=True,
        )

#########################################################################################################
architecture = ResNet1D(
    in_channels=2,
    layers=[2, 2],
    classes=1,
    kernel_size=3,
).to(device)

max_fpr = 1e-3
metric = BinaryAUROC(max_fpr=max_fpr)

model = Ml4gwDetectionModel(
    architecture=architecture,
    metric=metric,
)
#########################################################################################################
log_dir = data_dir / "logs"

logger = pl.loggers.CSVLogger(log_dir, name="ml4gw-expt")
trainer = pl.Trainer(
    max_epochs=30,
    precision="16-mixed",
    log_every_n_steps=5,
    logger=logger,
    callbacks=[pl.callbacks.RichProgressBar()],
    accelerator="gpu",
)
trainer.fit(model)
#########################################################################################################
import csv

path = log_dir / Path("ml4gw-expt")
# Take the most recent run, if we've done multiple
versions = [int(str(dir).split("_")[-1]) for dir in path.iterdir()]
version = sorted(versions)[-1]

with open(path / f"version_{version}/metrics.csv", newline="") as f:
    reader = csv.reader(f, delimiter=",")
    train_steps, train_loss, valid_steps, valid_loss = [], [], [], []
    _ = next(reader)
    for row in reader:
        if row[2] != "":
            train_steps.append(int(row[1]))
            train_loss.append(float(row[2]))
        else:
            valid_steps.append(int(row[1]))
            valid_loss.append(float(row[3]))
#########################################################################################################
plt.plot(train_steps, train_loss, label="Train loss")
plt.plot(valid_steps, valid_loss, label="Validation AUROC")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Metric value")













