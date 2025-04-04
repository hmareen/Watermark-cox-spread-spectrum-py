# Watermark-cox-spread-spectrum-py
Unofficial python implementation of the video spread-spectrum watermarking method by [Cox et al., "Secure spread spectrum watermarking for multimedia"](https://doi.org/10.1109/83.650120).

You can run the method by the following command:
    python embed_ss_watermark.py -i "${original_yuv}" -o "${watermarked_yuv}" -w $width -hh $height -f $amount_of_frames -ws $watermark_size -s $seed -a $alpha -sc $skip_coefficients -t $type

The argument `type` can be set to `ss` or `ss2` for watermarking every frame separately (performing a 2D DCT transform), or to `ss3` for watermarking a full video sequence (performing a 3D DCT transform).

This is the implementation used in the evaluation of [Mareen et al., "Implementation-Free Forensic Watermarking for Adaptive Streaming with A/B Watermarking"](https://doi.org/10.1007/978-981-16-2377-6_31) ([open-access version](https://www.researchgate.net/publication/354811387_Implementation-Free_Forensic_Watermarking_for_Adaptive_Streaming_with_AB_Watermarking)).
In that evaluation, the watermark length, the scale factor, and the number of skipped coefficients are 1000, 0.1, and 1000, respectively. The length and scale factor are the same that were used for the evaluation of the originally-proposed method by Cox et al. Additionally, the number of skipped coefficients is chosen to be as large as the length of the watermark, as was done in the evaluation of the adaptation by [Barni et al., "A DCT-domain system for robust image watermarking"](https://doi.org/10.1016/S0165-1684(98)00015-2).
