import { useEffect, useRef, type RefObject } from "react";
import { FilesetResolver, ImageSegmenter } from "@mediapipe/tasks-vision";

type CameraBackgroundProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  backgroundSource: string;
  onReadyChange: (ready: boolean) => void;
};

function drawCover(
  context: CanvasRenderingContext2D,
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  mirrored = false,
) {
  const scale = Math.max(
    targetWidth / sourceWidth,
    targetHeight / sourceHeight,
  );
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const x = (targetWidth - drawWidth) / 2;
  const y = (targetHeight - drawHeight) / 2;

  if (mirrored) {
    context.save();
    context.translate(targetWidth, 0);
    context.scale(-1, 1);
  }
  context.drawImage(source, x, y, drawWidth, drawHeight);
  if (mirrored) context.restore();
}

export function CameraBackground({
  videoRef,
  backgroundSource,
  onReadyChange,
}: CameraBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backgroundImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const background = new Image();
    background.src = backgroundSource;
    void background
      .decode()
      .then(() => {
        if (!cancelled) backgroundImageRef.current = background;
      })
      .catch(() => {
        if (!cancelled) backgroundImageRef.current = null;
      });

    return () => {
      cancelled = true;
    };
  }, [backgroundSource]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    let cancelled = false;
    let frameId = 0;
    let lastSegmentationAt = -Infinity;
    let segmenter: ImageSegmenter | null = null;
    let foregroundCanvas: HTMLCanvasElement | null = null;
    let foregroundContext: CanvasRenderingContext2D | null = null;
    let matteCanvas: HTMLCanvasElement | null = null;
    let matteContext: CanvasRenderingContext2D | null = null;
    let matteImage: ImageData | null = null;
    let isReady = false;

    const setReady = (ready: boolean) => {
      if (isReady !== ready) {
        isReady = ready;
        onReadyChange(ready);
      }
    };

    const drawFrame = (now: number) => {
      if (cancelled) return;
      frameId = window.requestAnimationFrame(drawFrame);

      const bounds = canvas.getBoundingClientRect();
      if (
        bounds.width === 0 ||
        bounds.height === 0 ||
        !segmenter ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0
      ) {
        return;
      }

      const background = backgroundImageRef.current;
      if (!background) return;

      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const backingWidth = Math.round(bounds.width * pixelRatio);
      const backingHeight = Math.round(bounds.height * pixelRatio);
      if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
        canvas.width = backingWidth;
        canvas.height = backingHeight;
      }
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      if (now - lastSegmentationAt >= 50) {
        lastSegmentationAt = now;

        try {
          const result = segmenter.segmentForVideo(video, now);
          const confidenceMask = result.confidenceMasks?.[0];
          if (confidenceMask) {
            const maskWidth = confidenceMask.width;
            const maskHeight = confidenceMask.height;
            if (
              !foregroundCanvas ||
              foregroundCanvas.width !== maskWidth ||
              foregroundCanvas.height !== maskHeight
            ) {
              foregroundCanvas = document.createElement("canvas");
              foregroundCanvas.width = maskWidth;
              foregroundCanvas.height = maskHeight;
              foregroundContext = foregroundCanvas.getContext("2d");
              matteCanvas = document.createElement("canvas");
              matteCanvas.width = maskWidth;
              matteCanvas.height = maskHeight;
              matteContext = matteCanvas.getContext("2d");
              matteImage = new ImageData(maskWidth, maskHeight);
            }

            const values = confidenceMask.getAsFloat32Array();
            if (
              foregroundContext &&
              matteCanvas &&
              matteContext &&
              matteImage
            ) {
              const pixels = matteImage.data;
              for (let pixel = 0; pixel < values.length; pixel += 1) {
                const offset = pixel * 4;
                const confidence = Math.max(
                  0,
                  Math.min(1, (values[pixel] - 0.18) / 0.62),
                );
                pixels[offset] = 255;
                pixels[offset + 1] = 255;
                pixels[offset + 2] = 255;
                pixels[offset + 3] = Math.round(confidence * 255);
              }
              matteContext.putImageData(matteImage, 0, 0);
              foregroundContext.globalCompositeOperation = "source-over";
              foregroundContext.clearRect(0, 0, maskWidth, maskHeight);
              foregroundContext.drawImage(video, 0, 0, maskWidth, maskHeight);
              foregroundContext.globalCompositeOperation = "destination-in";
              foregroundContext.drawImage(matteCanvas, 0, 0);
              foregroundContext.globalCompositeOperation = "source-over";
              setReady(true);
            }
          }
          result.close();
        } catch {
          setReady(false);
          context.clearRect(0, 0, bounds.width, bounds.height);
          return;
        }
      }

      if (!foregroundCanvas || !isReady) return;
      drawCover(
        context,
        background,
        background.naturalWidth,
        background.naturalHeight,
        bounds.width,
        bounds.height,
      );
      drawCover(
        context,
        foregroundCanvas,
        foregroundCanvas.width,
        foregroundCanvas.height,
        bounds.width,
        bounds.height,
        true,
      );
    };

    const initialize = async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks(
          `${import.meta.env.BASE_URL}mediapipe/wasm`,
        );
        if (cancelled) return;

        try {
          segmenter = await ImageSegmenter.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/selfie_segmenter.tflite`,
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        } catch {
          if (cancelled) return;
          segmenter = await ImageSegmenter.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/selfie_segmenter.tflite`,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        }

        if (cancelled) {
          segmenter.close();
          segmenter = null;
          return;
        }

        frameId = window.requestAnimationFrame(drawFrame);
      } catch {
        setReady(false);
      }
    };

    void initialize();
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frameId);
      segmenter?.close();
      context.clearRect(0, 0, canvas.width, canvas.height);
      setReady(false);
    };
  }, [onReadyChange, videoRef]);

  return (
    <canvas ref={canvasRef} className="camera-background" aria-hidden="true" />
  );
}
