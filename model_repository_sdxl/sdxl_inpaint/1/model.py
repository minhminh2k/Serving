import inspect
import logging

from typing import Optional, Any, Literal, Tuple, List
from pathlib import Path
from typing import Dict, List, Union

import json
import numpy as np
import torch
from transformers import CLIPTokenizer
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TritonPythonModel:
    tokenizer: CLIPTokenizer
    device: str
    scheduler: Union[
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
    ]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    eta: float

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        
        self.cache_dir = "/huggingface_cached"
        
        self.stable_diffusion_path = "/models/sdxl_inpaint"
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.stable_diffusion_path + "/1/tokenizer/",
            cache_dir=self.cache_dir
        )

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.stable_diffusion_path + "/1/tokenizer_2/",
            cache_dir=self.cache_dir
        )

        self.scheduler_config_path = self.stable_diffusion_path + "/1/scheduler/"
        self.scheduler = EulerDiscreteScheduler.from_config(self.scheduler_config_path)

        self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        self.height = 1024 # = self.unet.config.sample_size (128) * self.vae_scale_factor (8)
        self.width = 1024
        self.num_inference_steps = 20
        self.guidance_scale = 8.0
        self.strength = 0.999
        self.eta = 0.0
        self.scaling_factor = 0.13025
        self.text_encoder_projection_dim = 1280
        self.vae_latent_channels = 4
        self.unet_in_channels = 9
        self.vae_force_upcast = False


    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            
            image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            image = image_tensor.as_numpy()

            mask_tensor = pb_utils.get_input_tensor_by_name(request, "MASK_IMAGE")
            mask_image = mask_tensor.as_numpy()

            prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]
            
            logging.error(f"PROMPTTTTTT {prompt}")
            
            
            negative_prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                .as_numpy()
                .tolist()
            ]
            num_images_per_prompt = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                .as_numpy()
                .tolist()
            ][0]
            scheduler = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "SCHEDULER")
                .as_numpy()
                .tolist()
            ][0]
            if scheduler.__class__.__name__ != scheduler:
                self.scheduler = eval(
                    f"{scheduler}.from_config(self.scheduler_config_path)"
                )
            self.num_inference_steps = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]
            self.guidance_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            seed = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0]

            batch_size = len(prompt)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            ## Fix later
            if negative_prompt[0] == "NONE":
                negative_prompt = None
            
            # Encode Prompt
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer_2 is not None else [self.tokenizer]
            text_encoders = ["text_encoder", "text_encoder_2"] if self.tokenizer_2 is not None else ["text_encoder"]
            
            prompt_2 = prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logging.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                input_ids = text_input_ids.type(dtype=torch.int32)
                inputs = [
                    pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
                ]
                
                requested_output_name = ["last_hidden_state", "pooler_output", "hidden_states"] if text_encoder == "text_encoder" \
                    else ["text_embeds", "last_hidden_state", "hidden_states"]

                inference_request = pb_utils.InferenceRequest(
                    model_name=text_encoder,
                    requested_output_names=requested_output_name,
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    if text_encoder == "text_encoder":
                        output = pb_utils.get_output_tensor_by_name(
                            inference_response, "last_hidden_state"
                        )
                        output_pooler = pb_utils.get_output_tensor_by_name(
                            inference_response, "pooler_output"
                        )
                        output_hidden_states = pb_utils.get_output_tensor_by_name(
                            inference_response, "hidden_states"
                        )
                        pooled_prompt_embeds = torch.from_dlpack(output.to_dlpack())
                        
                        prompt_embeds: torch.Tensor = torch.from_dlpack(output_hidden_states.to_dlpack())
                    else:
                        output = pb_utils.get_output_tensor_by_name(
                            inference_response, "text_embeds"
                        )
                        output_pooler = pb_utils.get_output_tensor_by_name(
                            inference_response, "last_hidden_state"
                        )
                        output_hidden_states = pb_utils.get_output_tensor_by_name(
                            inference_response, "hidden_states"
                        )
                        pooled_prompt_embeds = torch.from_dlpack(output.to_dlpack())
                        prompt_embeds: torch.Tensor = torch.from_dlpack(output_hidden_states.to_dlpack())

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            do_classifier_free_guidance = self.guidance_scale > 1.0

            zero_out_negative_prompt = negative_prompt is None or negative_prompt == "NONE"
            if do_classifier_free_guidance and zero_out_negative_prompt:
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt
            
                # normalize str to list
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_2 = (
                    batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
                )

                uncond_tokens: List[str]
                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = [negative_prompt, negative_prompt_2]

                negative_prompt_embeds_list = []
                for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    # negative_prompt_embeds = text_encoder(
                    #     uncond_input.input_ids.to(self.device),
                    #     output_hidden_states=True,
                    # )

                    input_ids = uncond_input.input_ids.type(dtype=torch.int32)
                    inputs = [
                        pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
                    ]
                    
                    requested_output_name = ["last_hidden_state", "pooler_output", "hidden_states"] if text_encoder == "text_encoder" \
                        else ["text_embeds", "last_hidden_state", "hidden_states"]

                    inference_request = pb_utils.InferenceRequest(
                        model_name=text_encoder,
                        requested_output_names=requested_output_name,
                        inputs=inputs,
                    )
                    inference_response = inference_request.exec()
                    if inference_response.has_error():
                        raise pb_utils.TritonModelException(
                            inference_response.error().message()
                        )
                    else:                        
                        if text_encoder == "text_encoder":
                            output = pb_utils.get_output_tensor_by_name(
                                inference_response, "last_hidden_state"
                            )
                            
                            output_hidden_states = pb_utils.get_output_tensor_by_name(
                                inference_response, "hidden_states"
                            )
                        else:
                            output = pb_utils.get_output_tensor_by_name(
                                inference_response, "text_embeds"
                            )
                            output_hidden_states = pb_utils.get_output_tensor_by_name(
                                inference_response, "hidden_states"
                            )
                        
                    negative_pooled_prompt_embeds: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                    negative_prompt_embeds = torch.from_dlpack(output_hidden_states.to_dlpack())

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

                prompt_embeds = prompt_embeds.to(self.device, dtype=torch.float16)

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.float16, device=self.device)
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

                pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )

                # Not using unscale lora layers: USE_PEFT_BACKEND = False
                # return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            init_timestep = min(int(self.num_inference_steps * self.strength), self.num_inference_steps)
            t_start = max(self.num_inference_steps - init_timestep, 0)
            timesteps, num_inference_steps = timesteps, self.num_inference_steps - t_start

            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = self.strength == 1.0

            # 5. Preprocess mask and image
            init_image = self.image_processor.preprocess(image, height=self.height, width=self.width)
            init_image = init_image.to(dtype=torch.float32)
            
            mask_image = Image.fromarray(mask_image.squeeze().astype(np.uint8))
            mask = self.mask_processor.preprocess(mask_image, height=self.height, width=self.width)
            test_mask = mask.clone() # 1, 3, 1024, 1024
            
            if init_image.shape[1] == 4:
                # if images are in latent space, we can't mask it
                masked_image = None
            else:
                masked_image = init_image * (mask < 0.5)

            # 6. Prepare latent variables
            num_channels_latents = self.vae_latent_channels
            num_channels_unet = self.unet_in_channels
            return_image_latents = num_channels_unet == 4

            add_noise = True
            latents = None
            latents_outputs = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator,
                latents,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                add_noise=add_noise,
                return_noise=True,
                return_image_latents=return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs
                

            mask, masked_image_latents = self.prepare_mask_latents(
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Check that sizes of mask, masked image and latents match
            if num_channels_unet == 9:
                # default case for runwayml/stable-diffusion-inpainting
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = masked_image_latents.shape[1]
                if num_channels_latents + num_channels_mask + num_channels_masked_image != 9: # self.unet.config.in_channels:
                    raise ValueError(
                        f"{mask.shape}, {masked_image_latents.shape}, {test_mask.shape} Incorrect configuration settings! The config of `pipeline.unet`: expects"
                        f" {9} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input. "
                    )
            elif num_channels_unet != 4:
                raise ValueError(
                    f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
                )

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)

            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            height, width = latents.shape[-2:]
            height = height * self.vae_scale_factor
            width = width * self.vae_scale_factor

            original_size = None
            target_size = None
            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            negative_original_size = None
            negative_target_size = None
            # 10. Prepare added time ids & embeddings
            if negative_original_size is None:
                negative_original_size = original_size
            if negative_target_size is None:
                negative_target_size = target_size
            
            add_text_embeds = pooled_prompt_embeds
            text_encoder_projection_dim = self.text_encoder_projection_dim

            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size=original_size,
                crops_coords_top_left=(0,0),
                target_size=target_size,
                aesthetic_score=6.0,
                negative_aesthetic_score=2.5,
                negative_original_size=negative_original_size,
                negative_crops_coords_top_left=(0, 0),
                negative_target_size=negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(self.device)
            add_text_embeds = add_text_embeds.to(self.device)
            add_time_ids = add_time_ids.to(self.device)

            # 11. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 11.1 Optionally get Guidance Scale Embedding
            timestep_cond = None
            self._num_timesteps = len(timesteps)
            
            # Check from here
            
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                inputs = [
                    pb_utils.Tensor.from_dlpack(
                        "sample", torch.to_dlpack(latent_model_input)
                    ),
                    pb_utils.Tensor.from_dlpack(
                        "timestep", torch.to_dlpack(t.to(torch.float16).unsqueeze(0))),
                    pb_utils.Tensor.from_dlpack(
                        "encoder_hidden_states", torch.to_dlpack(prompt_embeds)
                    ),
                    pb_utils.Tensor.from_dlpack(
                        "text_embeds", torch.to_dlpack(add_text_embeds.to(torch.float16))
                    ),
                    pb_utils.Tensor.from_dlpack(
                        "time_ids", torch.to_dlpack(add_time_ids.to(torch.float16))
                    ),
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="unet",
                    requested_output_names=["out_sample"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "out_sample"
                    )
                    noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

            # scale and decode the image latents with vae
            latents = latents / self.scaling_factor
            
            latents = latents.type(dtype=torch.float32)
            
            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "latent_sample", torch.to_dlpack(latents.to(torch.float16))
                ),
            ]
            
            inference_request = pb_utils.InferenceRequest(
                model_name="vae_decoder",
                requested_output_names=["sample"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "sample")
                image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                image = image.type(dtype=torch.float32)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()

            tensor_output = [pb_utils.Tensor("IMAGES", image)]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
    
    def prepare_latents(
        self, 
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: str,
        generator: torch.Generator,
        latents: torch.Tensor = None,
        image: torch.Tensor = None,
        timestep: torch.Tensor = None,
        is_strength_max: bool = False,
        add_noise: bool = True,
        return_noise: bool = False,
        return_image_latents: bool = False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None and add_noise:
            noise = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.vae_force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                self.inference_triton_vae_latents(image[i : i + 1], generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.inference_triton_vae_latents(image, generator=generator)

        image_latents = image_latents.to(dtype)
        image_latents = self.scaling_factor * image_latents

        return image_latents
    
    def inference_triton_vae_latents(
        self,
        image: Any,
        generator: Any,
        return_dict: bool = False,
    ):
        inputs = [
            pb_utils.Tensor.from_dlpack(
                "sample", torch.to_dlpack(image.contiguous())
            )        
        ]

        inference_request = pb_utils.InferenceRequest(
            model_name="vae_encoder",
            requested_output_names=["latent_sample"],
            inputs=inputs,
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        else:
            output = pb_utils.get_output_tensor_by_name(
                inference_response, "latent_sample"
            )
            latents: torch.Tensor = torch.from_dlpack(output.to_dlpack())

        return latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            256 * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = 2816 # self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == 256 # self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == 256 # self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids