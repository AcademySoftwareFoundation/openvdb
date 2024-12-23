from garfield.garfield_datamanager import GarfieldDataManagerConfig
from garfield.garfield_field import GarfieldFieldConfig
from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipelineConfig
from garfield.garfield_model import GarfieldModelConfig
from garfield.garfield_pipeline import GarfieldPipelineConfig
from garfield.garfield_pixel_sampler import GarfieldPixelSamplerConfig
from garfield.img_group_model import ImgGroupModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

# For Gaussian Splatting
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

garfield_method = MethodSpecification(
    config=TrainerConfig(
        method_name="garfield",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=GarfieldPipelineConfig(
            datamanager=GarfieldDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                pixel_sampler=GarfieldPixelSamplerConfig(
                    num_rays_per_image=256,  # 4096/256 = 16 images per batch
                ),
                img_group_model=ImgGroupModelConfig(
                    model_type="sam_hf",
                    # Can choose out of "sam_fb", "sam_hf", "maskformer"
                    # Used sam_fb for the paper, see `img_group_model.py`.
                    device="cuda",
                ),
            ),
            model=GarfieldModelConfig(instance_field=GarfieldFieldConfig(n_instance_dims=256)),  # 256 in original
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "garfield": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15, weight_decay=1e-6, max_norm=1.0),
                # TODO the warmup_steps == pipeline.start_grouping_step, but would be good to not hardcode it
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=10000, warmup_steps=2000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Group Anything with Radiance Fields",
)

garfield_gauss_method = MethodSpecification(
    config=TrainerConfig(
        method_name="garfield-gauss",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=GarfieldGaussianPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=SplatfactoModelConfig(
                cull_alpha_thresh=0.2,
                use_scale_regularization=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="anythingnerf with gauss",
)
