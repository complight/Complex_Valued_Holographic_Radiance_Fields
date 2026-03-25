import torch
import numpy as np
import logging
from odak.learn.wave import get_propagation_kernel, custom
from odak.learn.wave import wavenumber, generate_complex_field, calculate_amplitude, calculate_phase
from odak.learn.tools import zero_pad, crop_center, circular_binary_mask, generate_2d_gaussian

class multiplane_loss_odak():
    """
    Loss function for computing loss in multiplanar images. Unlike, previous methods, this loss function accounts for defocused parts of an image.
    """

    def __init__(self, target_image, target_depth, blur_ratio = 0.25, 
                 target_blur_size = 10, number_of_planes = 4, weights = [1., 2.1, 0.6], 
                 multiplier = 1., scheme = 'defocus', reduction = 'mean', split_ratio = 1.0, device = torch.device('cpu')):
        """
        Parameters
        ----------
        target_image      : torch.tensor
                            Color target image [3 x m x n].
        target_depth      : torch.tensor
                            Monochrome target depth, same resolution as target_image.
        target_blur_size  : int
                            Maximum target blur size.
        blur_ratio        : float
                            Blur ratio, a value between zero and one.
        number_of_planes  : int
                            Number of planes.
        weights           : list
                            Weights of the loss function.
        multiplier        : float
                            Multiplier to multipy with targets.
        scheme            : str
                            The type of the loss, `naive` without defocus or `defocus` with defocus.
        reduction         : str
                            Reduction can either be 'mean', 'none' or 'sum'. For more see: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        device            : torch.device
                            Device to be used (e.g., cuda, cpu, opencl).
        """
        self.device = device
        self.target_image     = target_image.float().to(self.device)
        self.target_depth     = target_depth.float().to(self.device)
        self.target_blur_size = target_blur_size
        if self.target_blur_size % 2 == 0:
            self.target_blur_size += 1
        self.number_of_planes = number_of_planes
        self.multiplier       = multiplier
        self.weights          = weights
        self.reduction        = reduction
        self.blur_ratio       = blur_ratio
        self.split_ratio      = split_ratio
        self.set_targets()
        if scheme == 'defocus':
            self.add_defocus_blur()
        self.loss_function = torch.nn.MSELoss(reduction = self.reduction)
        
    def get_targets(self):
        """
        Returns
        -------
        targets           : torch.tensor
                            Returns a copy of the targets.
        target_depth      : torch.tensor
                            Returns a copy of the normalized quantized depth map.

        """
        divider = self.number_of_planes - 1
        if divider == 0:
            divider = 1
        return self.targets.detach().clone(), self.masks.detach().clone(), self.target_depth.detach().clone() / divider


    def set_targets(self):
        """
        Internal function for slicing the depth into planes without considering defocus. Users can query the results with get_targets() within the same class.
        """
        # self.target_depth = self.target_depth * (self.number_of_planes - 1)
        # self.target_depth = torch.round(self.target_depth, decimals = 0)

        normalized_depth = self.target_depth
        if self.number_of_planes == 2:
            # just for referennce, the emperically defined depth splitting ratio 
            # lego = 2.2, chicken = 1.6, mic = 1.6 fern = 1.0 ship = 1.6 chair = 2.2
            ratio = self.split_ratio
        else:
            ratio = 1.0
        biased_depth = torch.pow(normalized_depth, ratio) # small ratio, biased to front plane, large ratio biased to rear plane
        biased_depth = biased_depth * (self.number_of_planes - 1)
        self.target_depth = torch.round(biased_depth, decimals=0)
    
        self.targets      = torch.zeros(
                                        self.number_of_planes,
                                        self.target_image.shape[0],
                                        self.target_image.shape[1],
                                        self.target_image.shape[2],
                                        requires_grad = False,
                                        device = self.device
                                       )
        self.focus_target = torch.zeros_like(self.target_image, requires_grad = False)
        self.masks        = torch.zeros_like(self.targets)
        for i in range(self.number_of_planes):
            for ch in range(self.target_image.shape[0]):
                mask_zeros = torch.zeros_like(self.target_image[ch], dtype = torch.int)
                mask_ones = torch.ones_like(self.target_image[ch], dtype = torch.int)
                mask = torch.where(self.target_depth == i, mask_ones, mask_zeros)
                new_target = self.target_image[ch] * mask
                self.focus_target = self.focus_target + new_target.squeeze(0).squeeze(0).detach().clone()
                self.targets[i, ch] = new_target.squeeze(0).squeeze(0)
                self.masks[i, ch] = mask.detach().clone() 


    def add_defocus_blur(self):
        """
        Internal function for adding defocus blur to the multiplane targets. Users can query the results with get_targets() within the same class.
        """
        kernel_length = [self.target_blur_size, self.target_blur_size ]
        for ch in range(self.target_image.shape[0]):
            targets_cache = self.targets[:, ch].detach().clone()
            target = torch.sum(targets_cache, axis = 0)
            for i in range(self.number_of_planes):
                defocus = torch.zeros_like(targets_cache[i])
                for j in range(self.number_of_planes):
                    nsigma = [int(abs(i - j) * self.blur_ratio), int(abs(i -j) * self.blur_ratio)]
                    if torch.sum(targets_cache[j]) > 0:
                        if i == j:
                            nsigma = [0., 0.]
                        kernel = generate_2d_gaussian(kernel_length, nsigma).to(self.device)
                        kernel = kernel / torch.sum(kernel)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)
                        target_current = target.detach().clone().unsqueeze(0).unsqueeze(0)
                        defocus_plane = torch.nn.functional.conv2d(target_current, kernel, padding = 'same')
                        defocus_plane = defocus_plane.view(defocus_plane.shape[-2], defocus_plane.shape[-1])
                        defocus = defocus + defocus_plane * torch.abs(self.masks[j, ch])
                self.targets[i, ch] = defocus
        self.targets = self.targets.detach().clone() * self.multiplier
    

    def __call__(self, image, target, plane_id = None, inject_noise = False, noise_ratio = 1e-3):
        """
        Calculates the multiplane loss against a given target.
        
        Parameters
        ----------
        image         : torch.tensor
                        Image to compare with a target [3 x m x n].
        target        : torch.tensor
                        Target image for comparison [3 x m x n].
        plane_id      : int
                        Number of the plane under test.
        inject_noise  : bool
                        When True, noise is added on the targets at the given `noise_ratio`.
        noise_ratio   : float
                        Noise ratio.
        
        Returns
        -------
        loss          : torch.tensor
                        Computed loss.
        """
        l2 = self.weights[0] * self.loss_function(image, target)
        if isinstance(plane_id, type(None)):
            mask = self.masks
        else:
            mask= self.masks[plane_id, :]
        if inject_noise:
            target = target + torch.randn_like(target) * noise_ratio * (target.max() - target.min())
        l2_mask = self.weights[1] * self.loss_function(image * mask, target * mask)
        l2_cor = self.weights[2] * self.loss_function(image * target, target * target)
        loss = l2 + l2_mask + l2_cor
        return loss

class propagator():
    """
    A light propagation model that propagates light to desired image plane with two separate propagations. 
    We use this class in our various works including `Kavaklı et al., Realistic Defocus Blur for Multiplane Computer-Generated Holography`.
    """
    def __init__(
                 self,
                 resolution = [1920, 1080],
                 wavelengths = [515e-9,],
                 pixel_pitch = 8e-6,
                 resolution_factor = 1,
                 number_of_frames = 1,
                 number_of_depth_layers = 1,
                 volume_depth = 1e-2,
                 image_location_offset = 5e-3,
                 propagation_type = 'Bandlimited Angular Spectrum',
                 propagator_type = 'back and forth',
                 back_and_forth_distance = 0.3,
                 laser_channel_power = None,
                 aperture = None,
                 aperture_size = None,
                 distances = None,
                 aperture_samples = [20, 20, 5, 5],
                 method = 'conventional',
                 device = torch.device('cpu')
                ):
        """
        Parameters
        ----------
        resolution              : list
                                  Resolution.
        wavelengths             : float
                                  Wavelength of light in meters.
        pixel_pitch             : float
                                  Pixel pitch in meters.
        resolution_factor       : int
                                  Resolution factor for scaled simulations.
        number_of_frames        : int
                                  Number of hologram frames.
                                  Typically, there are three frames, each one for a single color primary.
        number_of_depth_layers  : int
                                  Equ-distance number of depth layers within the desired volume. If `distances` parameter is passed, this value will be automatically set to the length of the `distances` verson provided.
        volume_depth            : float
                                  Width of the volume along the propagation direction.
        image_location_offset   : float
                                  Center of the volume along the propagation direction.
        propagation_type        : str
                                  Propagation type. 
                                  See ropagate_beam() and odak.learn.wave.get_propagation_kernel() for more.
        propagator_type         : str
                                  Propagator type.
                                  The options are `back and forth` and `forward` propagators.
        back_and_forth_distance : float
                                  Zero mode distance for `back and forth` propagator type.
        laser_channel_power     : torch.tensor
                                  Laser channel powers for given number of frames and number of wavelengths.
        aperture                : torch.tensor
                                  Aperture at the Fourier plane.
        aperture_size           : float
                                  Aperture width for a circular aperture.
        aperture_samples        : list
                                  When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for hologram plane pixel and the last two is for image plane pixel.
        distances               : torch.tensor
                                  Propagation distances in meters.
        method                  : str
                                  Hologram type conventional or multi-color.
        device                  : torch.device
                                  Device to be used for computation. For more see torch.device().
        """
        self.device = device
        self.pixel_pitch = pixel_pitch
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.propagation_type = propagation_type
        if self.propagation_type != 'Impulse Response Fresnel':
            resolution_factor = 1
        self.resolution_factor = resolution_factor
        self.number_of_frames = number_of_frames
        self.number_of_depth_layers = number_of_depth_layers
        self.number_of_channels = len(self.wavelengths)
        self.volume_depth = volume_depth
        self.image_location_offset = image_location_offset
        self.propagator_type = propagator_type
        self.aperture_samples = aperture_samples
        self.zero_mode_distance = torch.tensor(back_and_forth_distance, device = device)
        self.method = method
        self.aperture = aperture
        self.init_distances(distances)
        self.init_kernels()
        self.init_channel_power(laser_channel_power)
        self.init_phase_scale()
        self.set_aperture(aperture, aperture_size)

    def init_distances(self, distances):
        """
        Internal function to initialize distances.

        Parameters
        ----------
        distances               : torch.tensor
                                  Propagation distances.
        """
        if isinstance(distances, type(None)):
            self.distances = torch.linspace(-self.volume_depth / 2., self.volume_depth / 2., self.number_of_depth_layers) + self.image_location_offset
        else:
            self.distances = torch.as_tensor(distances)
            self.number_of_depth_layers = self.distances.shape[0]
        logging.warning('Distances: {}'.format(self.distances))


    def init_kernels(self):
        """
        Internal function to initialize kernels.
        """
        self.generated_kernels = torch.zeros(
                                             self.number_of_depth_layers,
                                             self.number_of_channels,
                                             device = self.device
                                            )
        self.kernels = torch.zeros(
                                   self.number_of_depth_layers,
                                   self.number_of_channels,
                                   self.resolution[0] * self.resolution_factor * 2,
                                   self.resolution[1] * self.resolution_factor * 2,
                                   dtype = torch.complex64,
                                   device = self.device
                                  )


    def init_channel_power(self, channel_power):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        self.channel_power = channel_power
        if isinstance(self.channel_power, type(None)):
            self.channel_power = torch.eye(
                                           self.number_of_frames,
                                           self.number_of_channels,
                                           device = self.device,
                                           requires_grad = False
                                          )


    def init_phase_scale(self):
        """
        Internal function to set the phase scale.
        In some cases, you may want to modify this init to ratio phases for different color primaries as an SLM is configured for a specific central wavelength.
        """
        self.phase_scale = torch.tensor(
                                        [
                                         1.,
                                         1.,
                                         1.
                                        ],
                                        requires_grad = False,
                                        device = self.device
                                       )


    def set_aperture(self, aperture = None, aperture_size = None):
        """
        Set aperture in the Fourier plane.


        Parameters
        ----------
        aperture        : torch.tensor
                          Aperture at the original resolution of a hologram.
                          If aperture is provided as None, it will assign a circular aperture at the size of the short edge (width or height).
        aperture_size   : int
                          If no aperture is provided, this will determine the size of the circular aperture.
        """
        # print("odak prop set: ", aperture_size)
        if isinstance(aperture, type(None)):
            if isinstance(aperture_size, type(None)):
                aperture_size = torch.max(
                                          torch.tensor([
                                                        self.resolution[0] * self.resolution_factor, 
                                                        self.resolution[1] * self.resolution_factor
                                                       ])
                                         )
            self.aperture = circular_binary_mask(
                                                 self.resolution[0] * self.resolution_factor * 2,
                                                 self.resolution[1] * self.resolution_factor * 2,
                                                 aperture_size,
                                                ).to(self.device) * 1.
        else:
            self.aperture = zero_pad(aperture).to(self.device) * 1.


    def get_laser_powers(self):
        """
        Internal function to get the laser powers.

        Returns
        -------
        laser_power      : torch.tensor
                           Laser powers.
        """
        if self.method == 'conventional':
            laser_power = self.channel_power
        if self.method == 'multi-color':
            laser_power = torch.abs(torch.cos(self.channel_power))
        return laser_power


    def set_laser_powers(self, laser_power):
        """
        Internal function to set the laser powers.

        Parameters
        -------
        laser_power      : torch.tensor
                           Laser powers.
        """
        self.channel_power = laser_power



    def get_kernels(self):
        """
        Function to return the kernels used in the light transport.
        
        Returns
        -------
        kernels           : torch.tensor
                            Kernel amplitudes.
        """
        h = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(self.kernels)))
        kernels_amplitude = calculate_amplitude(h)
        kernels_phase = calculate_phase(h)
        return kernels_amplitude, kernels_phase


    def __call__(self, input_field, channel_id, depth_id):
        """
        Function that represents the forward model in hologram optimization.

        Parameters
        ----------
        input_field         : torch.tensor
                              Input complex input field.
        channel_id          : int
                              Identifying the color primary to be used.
        depth_id            : int
                              Identifying the depth layer to be used.

        Returns
        -------
        output_field        : torch.tensor
                              Propagated output complex field.
        """
        distance = self.distances[depth_id]
        if not self.generated_kernels[depth_id, channel_id]:
            if self.propagator_type == 'forward':
                H = get_propagation_kernel(
                                           nu = self.resolution[0] * 2,
                                           nv = self.resolution[1] * 2,
                                           dx = self.pixel_pitch,
                                           wavelength = self.wavelengths[channel_id],
                                           distance = distance,
                                           device = self.device,
                                           propagation_type = self.propagation_type,
                                           samples = self.aperture_samples,
                                           scale = self.resolution_factor
                                          )
            elif self.propagator_type == 'back and forth':
                H_forward = get_propagation_kernel(
                                                   nu = self.resolution[0] * 2,
                                                   nv = self.resolution[1] * 2,
                                                   dx = self.pixel_pitch,
                                                   wavelength = self.wavelengths[channel_id],
                                                   distance = self.zero_mode_distance,
                                                   device = self.device,
                                                   propagation_type = self.propagation_type,
                                                   samples = self.aperture_samples,
                                                   scale = self.resolution_factor
                                                  )
                distance_back = -(self.zero_mode_distance + self.image_location_offset - distance)
                H_back = get_propagation_kernel(
                                                nu = self.resolution[0] * 2,
                                                nv = self.resolution[1] * 2,
                                                dx = self.pixel_pitch,
                                                wavelength = self.wavelengths[channel_id],
                                                distance = distance_back,
                                                device = self.device,
                                                propagation_type = self.propagation_type,
                                                samples = self.aperture_samples,
                                                scale = self.resolution_factor
                                               )
                H = H_forward * H_back
            self.kernels[depth_id, channel_id] = H
            self.generated_kernels[depth_id, channel_id] = True
        else:
            H = self.kernels[depth_id, channel_id].detach().clone()
        field_scale = input_field
        field_scale_padded = zero_pad(field_scale)
        output_field_padded = custom(field_scale_padded, H, aperture = self.aperture)
        output_field = crop_center(output_field_padded)
        return output_field

    def _ensure_all_kernels_generated(self):
        """
        Ensures all kernels are pre-computed for all depths and channels.
        This improves performance by avoiding kernel generation during reconstruction.
        """
        for depth_id in range(self.number_of_depth_layers):
            distance = self.distances[depth_id]
            for channel_id in range(self.number_of_channels):
                if not self.generated_kernels[depth_id, channel_id]:
                    if self.propagator_type == 'forward':
                        H = get_propagation_kernel(
                            nu=self.resolution[0] * 2,
                            nv=self.resolution[1] * 2,
                            dx=self.pixel_pitch,
                            wavelength=self.wavelengths[channel_id],
                            distance=distance,
                            device=self.device,
                            propagation_type=self.propagation_type,
                            samples=self.aperture_samples,
                            scale=self.resolution_factor
                        )
                    elif self.propagator_type == 'back and forth':
                        H_forward = get_propagation_kernel(
                            nu=self.resolution[0] * 2,
                            nv=self.resolution[1] * 2,
                            dx=self.pixel_pitch,
                            wavelength=self.wavelengths[channel_id],
                            distance=self.zero_mode_distance,
                            device=self.device,
                            propagation_type=self.propagation_type,
                            samples=self.aperture_samples,
                            scale=self.resolution_factor
                        )
                        distance_back = -(self.zero_mode_distance + self.image_location_offset - distance)
                        H_back = get_propagation_kernel(
                            nu=self.resolution[0] * 2,
                            nv=self.resolution[1] * 2,
                            dx=self.pixel_pitch,
                            wavelength=self.wavelengths[channel_id],
                            distance=distance_back,
                            device=self.device,
                            propagation_type=self.propagation_type,
                            samples=self.aperture_samples,
                            scale=self.resolution_factor
                        )
                        H = H_forward * H_back
                    self.kernels[depth_id, channel_id] = H
                    self.generated_kernels[depth_id, channel_id] = True

    def reconstruct(self, hologram_phases, amplitude=None, no_grad=True, get_complex=False):
        """
        Optimized function to reconstruct a given hologram.

        Parameters
        ----------
        hologram_phases : torch.tensor
                         Hologram phases [ch x m x n].
        amplitude       : torch.tensor
                         Amplitude profiles for each color primary [ch x m x n]
        no_grad         : bool
                         If set True, uses torch.no_grad in reconstruction.
        get_complex     : bool
                         If set True, reconstructor returns the complex field but not the intensities.

        Returns
        -------
        reconstructions : torch.tensor
                         Reconstructed frames.
        """
        if no_grad:
            with torch.no_grad():
                return self._reconstruct_impl(hologram_phases, amplitude, get_complex)
        else:
            return self._reconstruct_impl(hologram_phases, amplitude, get_complex)

    def _reconstruct_impl(self, hologram_phases, amplitude=None, get_complex=False):
        """
        Internal implementation of reconstruction without redundant kernel generation.
        """
        # Handle dimensions
        if len(hologram_phases.shape) > 3:
            hologram_phases = hologram_phases.squeeze(0)
        
        # Set output type
        reconstruction_type = torch.complex64 if get_complex else torch.float32
        
        # Check frame count
        if hologram_phases.shape[0] != self.number_of_frames:
            logging.warning('Provided hologram frame count is {} but the configured number of frames is {}.'.format(
                hologram_phases.shape[0], self.number_of_frames))
        
        # Initialize reconstructions tensor
        reconstructions = torch.zeros(
            self.number_of_frames,
            self.number_of_depth_layers,
            self.number_of_channels,
            self.resolution[0] * self.resolution_factor,
            self.resolution[1] * self.resolution_factor,
            dtype=reconstruction_type,
            device=self.device
        )
        
        # Create amplitude if not provided
        if isinstance(amplitude, type(None)):
            amplitude = torch.zeros(
                self.number_of_channels,
                self.resolution[0] * self.resolution_factor,
                self.resolution[1] * self.resolution_factor,
                device=self.device
            )
            amplitude[:, ::self.resolution_factor, ::self.resolution_factor] = 1.
        
        # Handle resolution scaling
        if self.resolution_factor != 1:
            hologram_phases_scaled = torch.zeros(
                self.number_of_frames,
                self.resolution[0] * self.resolution_factor,
                self.resolution[1] * self.resolution_factor,
                device=self.device
            )
            hologram_phases_scaled[:, ::self.resolution_factor, ::self.resolution_factor] = hologram_phases
        else:
            hologram_phases_scaled = hologram_phases
        
        # Get laser powers
        laser_powers = self.get_laser_powers()
        
        # Pre-compute all kernels first (this avoids kernel generation inside the main loops)
        self._ensure_all_kernels_generated()
        
        # Main processing loop
        for frame_id in range(self.number_of_frames):
            phase = hologram_phases_scaled[frame_id]
            
            for depth_id in range(self.number_of_depth_layers):
                for channel_id in range(self.number_of_channels):
                    # Get laser power for this frame and channel
                    laser_power = laser_powers[frame_id][channel_id]
                    
                    # Generate hologram field with phase modulation
                    hologram = generate_complex_field(
                        laser_power * amplitude[channel_id],
                        phase * self.phase_scale[channel_id]
                    )
                    
                    # Get pre-computed kernel for this depth and channel
                    H = self.kernels[depth_id, channel_id]
                    
                    # Zero pad the hologram for FFT
                    field_scale_padded = zero_pad(hologram)
                    
                    # Apply the custom light propagation
                    output_field_padded = custom(field_scale_padded, H, aperture=self.aperture)
                    
                    # Crop to original size
                    output_field = crop_center(output_field_padded)
                    
                    # Store result (either complex field or intensity)
                    if get_complex:
                        reconstructions[frame_id, depth_id, channel_id] = output_field
                    else:
                        reconstructions[frame_id, depth_id, channel_id] = calculate_amplitude(output_field) ** 2
        
        return reconstructions

