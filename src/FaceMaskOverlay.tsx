import { useEffect, useRef, type RefObject } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

type FaceMaskOverlayProps = {
  videoRef: RefObject<HTMLVideoElement | null>;
  onStatus: (status: string) => void;
  maskSource: string;
};

type Point = { x: number; y: number };
type MaskPose = Point & { width: number; height: number; angle: number };
type FaceTrack = { id: number; pose: MaskPose; lastSeenAt: number };
type EyeOpening = {
  center: Point;
  width: number;
  height: number;
  iris: Point;
};
type FaceFeatures = {
  faceWidth: number;
  faceHeight: number;
  eyes: [EyeOpening, EyeOpening];
  mouth: { center: Point; width: number; gap: number };
  jawOpening: number;
  mouthOpening: number;
  mouthRoundness: number;
  mouthStretch: number;
};
type ZombieEyeSprites = {
  lookLeft: [HTMLCanvasElement, HTMLCanvasElement];
  lookRight: [HTMLCanvasElement, HTMLCanvasElement];
};
type ZombieJawSprites = {
  closed: HTMLCanvasElement;
  open: HTMLCanvasElement;
};
type SkullEyeSprites = {
  lookLeft: [HTMLCanvasElement, HTMLCanvasElement];
  lookCenter: [HTMLCanvasElement, HTMLCanvasElement];
  lookRight: [HTMLCanvasElement, HTMLCanvasElement];
  squintLeft: [HTMLCanvasElement, HTMLCanvasElement];
  squintCenter: [HTMLCanvasElement, HTMLCanvasElement];
  squintRight: [HTMLCanvasElement, HTMLCanvasElement];
};
type SkullJawSprites = {
  closed: HTMLCanvasElement;
  open: HTMLCanvasElement;
};
type PhantomEyeSprites = {
  lookLeft: [HTMLCanvasElement, HTMLCanvasElement];
  lookCenter: [HTMLCanvasElement, HTMLCanvasElement];
  lookRight: [HTMLCanvasElement, HTMLCanvasElement];
  blink: [HTMLCanvasElement, HTMLCanvasElement];
};
type MaskFeatureAnchors = {
  eyes: [Point, Point];
  eyeRadiusX: number;
  eyeRadiusY: number;
  mouth: Point;
  mouthRadiusX: number;
  mouthRadiusY: number;
  animateFeatures?: boolean;
  animateMouth?: boolean;
};

const MAX_TRACKED_FACES = 4;
const MASKS_TO_REMOVE_BACKGROUND = new Set([
  "Frankenstein.png",
  "doll.png",
  "doll2.png",
  "jack.png",
  "phantom.png",
  "pumpkin.png",
  "skull.png",
  "zombie.png",
]);
const FRANKENSTEIN_EYE_ANCHORS = [
  { x: 0.382, y: 0.493 },
  { x: 0.628, y: 0.493 },
] as const;
const FRANKENSTEIN_MOUTH_ANCHOR = { x: 0.5, y: 0.693 };
const MASK_FEATURE_ANCHORS: Record<string, MaskFeatureAnchors> = {
  "doll.png": {
    eyes: [
      { x: 0.34, y: 0.49 },
      { x: 0.66, y: 0.49 },
    ],
    eyeRadiusX: 0.085,
    eyeRadiusY: 0.048,
    mouth: { x: 0.5, y: 0.76 },
    mouthRadiusX: 0.17,
    mouthRadiusY: 0.045,
    animateFeatures: true,
  },
  "doll2.png": {
    eyes: [
      { x: 0.34, y: 0.49 },
      { x: 0.66, y: 0.49 },
    ],
    eyeRadiusX: 0.085,
    eyeRadiusY: 0.05,
    mouth: { x: 0.5, y: 0.76 },
    mouthRadiusX: 0.17,
    mouthRadiusY: 0.045,
    animateFeatures: true,
  },
  "jack.png": {
    eyes: [
      { x: 0.34, y: 0.44 },
      { x: 0.66, y: 0.44 },
    ],
    eyeRadiusX: 0.09,
    eyeRadiusY: 0.065,
    mouth: { x: 0.5, y: 0.66 },
    mouthRadiusX: 0.25,
    mouthRadiusY: 0.07,
    animateFeatures: true,
    animateMouth: false,
  },
  "phantom.png": {
    eyes: [
      { x: 0.37, y: 0.41 },
      { x: 0.63, y: 0.41 },
    ],
    eyeRadiusX: 0.09,
    eyeRadiusY: 0.052,
    mouth: { x: 0.5, y: 0.69 },
    mouthRadiusX: 0.14,
    mouthRadiusY: 0.09,
    animateFeatures: true,
  },
  "pumpkin.png": {
    eyes: [
      { x: 0.31, y: 0.42 },
      { x: 0.69, y: 0.42 },
    ],
    eyeRadiusX: 0.095,
    eyeRadiusY: 0.07,
    mouth: { x: 0.5, y: 0.68 },
    mouthRadiusX: 0.26,
    mouthRadiusY: 0.075,
    animateFeatures: true,
    animateMouth: false,
  },
  "skull.png": {
    eyes: [
      { x: 0.32, y: 0.43 },
      { x: 0.68, y: 0.43 },
    ],
    eyeRadiusX: 0.095,
    eyeRadiusY: 0.06,
    mouth: { x: 0.5, y: 0.7 },
    mouthRadiusX: 0.19,
    mouthRadiusY: 0.055,
    animateFeatures: true,
  },
  "specter.svg": {
    eyes: [
      { x: 0.35, y: 0.42 },
      { x: 0.65, y: 0.42 },
    ],
    eyeRadiusX: 0.095,
    eyeRadiusY: 0.06,
    mouth: { x: 0.5, y: 0.78 },
    mouthRadiusX: 0.16,
    mouthRadiusY: 0.06,
    animateFeatures: true,
  },
  "thorn.svg": {
    eyes: [
      { x: 0.34, y: 0.46 },
      { x: 0.66, y: 0.46 },
    ],
    eyeRadiusX: 0.09,
    eyeRadiusY: 0.052,
    mouth: { x: 0.5, y: 0.77 },
    mouthRadiusX: 0.17,
    mouthRadiusY: 0.055,
    animateFeatures: true,
  },
  "zombie.png": {
    eyes: [
      { x: 0.33, y: 0.41 },
      { x: 0.67, y: 0.41 },
    ],
    eyeRadiusX: 0.09,
    eyeRadiusY: 0.06,
    mouth: { x: 0.5, y: 0.71 },
    mouthRadiusX: 0.2,
    mouthRadiusY: 0.06,
    animateFeatures: true,
  },
};

function distance(first: Point, second: Point) {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function interpolateAngle(current: number, target: number, amount: number) {
  const difference = Math.atan2(
    Math.sin(target - current),
    Math.cos(target - current),
  );
  return current + difference * amount;
}

function removeEdgeBackground(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  darkBackgroundMax = 72,
  darkFeatherMax = 96,
) {
  const imageData = context.getImageData(0, 0, width, height);
  const pixels = imageData.data;
  const cornerIndexes = [
    0,
    width - 1,
    (height - 1) * width,
    height * width - 1,
  ];
  if (cornerIndexes.some((index) => pixels[index * 4 + 3] < 255)) return;

  const cornerBrightness = cornerIndexes.reduce((total, index) => {
    const offset = index * 4;
    return total + pixels[offset] + pixels[offset + 1] + pixels[offset + 2];
  }, 0);
  const lightBackground = cornerBrightness / (cornerIndexes.length * 3) > 127;
  const background = new Uint8Array(width * height);
  const queue = new Uint32Array(width * height);
  let queueStart = 0;
  let queueEnd = 0;

  const enqueueBackground = (x: number, y: number) => {
    const pixelIndex = y * width + x;
    if (background[pixelIndex]) return;

    const offset = pixelIndex * 4;
    const red = pixels[offset];
    const green = pixels[offset + 1];
    const blue = pixels[offset + 2];
    const minimumChannel = Math.min(red, green, blue);
    const maximumChannel = Math.max(red, green, blue);
    const channelSpread = maximumChannel - minimumChannel;
    const isBackground = lightBackground
      ? minimumChannel >= 215 && channelSpread <= 55
      : maximumChannel <= darkBackgroundMax && channelSpread <= 40;

    if (!isBackground) return;
    background[pixelIndex] = 1;
    queue[queueEnd++] = pixelIndex;
  };

  for (let x = 0; x < width; x += 1) {
    enqueueBackground(x, 0);
    enqueueBackground(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueueBackground(0, y);
    enqueueBackground(width - 1, y);
  }

  while (queueStart < queueEnd) {
    const pixelIndex = queue[queueStart++];
    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    if (x > 0) enqueueBackground(x - 1, y);
    if (x < width - 1) enqueueBackground(x + 1, y);
    if (y > 0) enqueueBackground(x, y - 1);
    if (y < height - 1) enqueueBackground(x, y + 1);
  }

  for (let pixelIndex = 0; pixelIndex < background.length; pixelIndex += 1) {
    const offset = pixelIndex * 4;
    if (background[pixelIndex]) {
      pixels[offset + 3] = 0;
      continue;
    }

    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    const touchesBackground =
      (x > 0 && background[pixelIndex - 1]) ||
      (x < width - 1 && background[pixelIndex + 1]) ||
      (y > 0 && background[pixelIndex - width]) ||
      (y < height - 1 && background[pixelIndex + width]);
    if (!touchesBackground) continue;

    const red = pixels[offset];
    const green = pixels[offset + 1];
    const blue = pixels[offset + 2];
    const minimumChannel = Math.min(red, green, blue);
    const maximumChannel = Math.max(red, green, blue);
    const channelSpread = maximumChannel - minimumChannel;
    const isNearBackground = lightBackground
      ? minimumChannel >= 200 && channelSpread <= 70
      : maximumChannel <= darkFeatherMax && channelSpread <= 60;
    if (!isNearBackground) continue;

    const opacity = lightBackground
      ? (255 - minimumChannel) / 255
      : maximumChannel / 255;
    pixels[offset + 3] = Math.round(pixels[offset + 3] * opacity);
  }

  context.putImageData(imageData, 0, 0);
}

function removeCheckerboardBackground(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
) {
  const imageData = context.getImageData(0, 0, width, height);
  const pixels = imageData.data;
  const background = new Uint8Array(width * height);
  const queue = new Uint32Array(width * height);
  let queueStart = 0;
  let queueEnd = 0;

  const enqueueBackground = (x: number, y: number) => {
    const pixelIndex = y * width + x;
    if (background[pixelIndex]) return;

    const offset = pixelIndex * 4;
    const red = pixels[offset];
    const green = pixels[offset + 1];
    const blue = pixels[offset + 2];
    const minimumChannel = Math.min(red, green, blue);
    const maximumChannel = Math.max(red, green, blue);
    if (minimumChannel < 165 || maximumChannel - minimumChannel > 32) return;

    background[pixelIndex] = 1;
    queue[queueEnd++] = pixelIndex;
  };

  for (let x = 0; x < width; x += 1) {
    enqueueBackground(x, 0);
    enqueueBackground(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueueBackground(0, y);
    enqueueBackground(width - 1, y);
  }

  while (queueStart < queueEnd) {
    const pixelIndex = queue[queueStart++];
    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    if (x > 0) enqueueBackground(x - 1, y);
    if (x < width - 1) enqueueBackground(x + 1, y);
    if (y > 0) enqueueBackground(x, y - 1);
    if (y < height - 1) enqueueBackground(x, y + 1);
  }

  for (let pixelIndex = 0; pixelIndex < background.length; pixelIndex += 1) {
    if (background[pixelIndex]) pixels[pixelIndex * 4 + 3] = 0;
  }

  context.putImageData(imageData, 0, 0);
}

function removeEdgeColorBackground(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  tolerance = 10,
  edgeTolerance = 16,
) {
  const imageData = context.getImageData(0, 0, width, height);
  const pixels = imageData.data;
  const cornerIndexes = [
    0,
    width - 1,
    (height - 1) * width,
    height * width - 1,
  ];
  const backgroundColor = [0, 1, 2].map((channel) =>
    Math.round(
      cornerIndexes.reduce(
        (total, pixelIndex) => total + pixels[pixelIndex * 4 + channel],
        0,
      ) / cornerIndexes.length,
    ),
  );
  const background = new Uint8Array(width * height);
  const queue = new Uint32Array(width * height);
  let queueStart = 0;
  let queueEnd = 0;

  const colorDistance = (pixelIndex: number) => {
    const offset = pixelIndex * 4;
    return Math.max(
      Math.abs(pixels[offset] - backgroundColor[0]),
      Math.abs(pixels[offset + 1] - backgroundColor[1]),
      Math.abs(pixels[offset + 2] - backgroundColor[2]),
    );
  };

  const enqueueBackground = (x: number, y: number) => {
    const pixelIndex = y * width + x;
    if (background[pixelIndex] || colorDistance(pixelIndex) > tolerance) return;
    background[pixelIndex] = 1;
    queue[queueEnd++] = pixelIndex;
  };

  for (let x = 0; x < width; x += 1) {
    enqueueBackground(x, 0);
    enqueueBackground(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueueBackground(0, y);
    enqueueBackground(width - 1, y);
  }

  while (queueStart < queueEnd) {
    const pixelIndex = queue[queueStart++];
    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    if (x > 0) enqueueBackground(x - 1, y);
    if (x < width - 1) enqueueBackground(x + 1, y);
    if (y > 0) enqueueBackground(x, y - 1);
    if (y < height - 1) enqueueBackground(x, y + 1);
  }

  for (let pixelIndex = 0; pixelIndex < background.length; pixelIndex += 1) {
    const offset = pixelIndex * 4;
    if (background[pixelIndex]) {
      pixels[offset + 3] = 0;
      continue;
    }

    const x = pixelIndex % width;
    const y = Math.floor(pixelIndex / width);
    const touchesBackground =
      (x > 0 && background[pixelIndex - 1]) ||
      (x < width - 1 && background[pixelIndex + 1]) ||
      (y > 0 && background[pixelIndex - width]) ||
      (y < height - 1 && background[pixelIndex + width]);
    const distance = colorDistance(pixelIndex);
    if (!touchesBackground || distance >= edgeTolerance) continue;

    pixels[offset + 3] = Math.round(
      (pixels[offset + 3] * (distance - tolerance)) /
        (edgeTolerance - tolerance),
    );
  }

  context.putImageData(imageData, 0, 0);
}

export function FaceMaskOverlay({
  videoRef,
  onStatus,
  maskSource,
}: FaceMaskOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const preparedMaskRef = useRef<HTMLCanvasElement | null>(null);
  const preparedMaskNameRef = useRef("");
  const preparedMaskScaleRef = useRef(1);
  const faceFeaturesRef = useRef<FaceFeatures | null>(null);
  const zombieEyeSpritesRef = useRef<ZombieEyeSprites | null>(null);
  const zombieJawSpritesRef = useRef<ZombieJawSprites | null>(null);
  const skullEyeSpritesRef = useRef<SkullEyeSprites | null>(null);
  const skullJawSpritesRef = useRef<SkullJawSprites | null>(null);
  const phantomEyeSpritesRef = useRef<PhantomEyeSprites | null>(null);

  useEffect(() => {
    let cancelled = false;
    zombieEyeSpritesRef.current = null;
    zombieJawSpritesRef.current = null;
    skullEyeSpritesRef.current = null;
    skullJawSpritesRef.current = null;
    phantomEyeSpritesRef.current = null;

    const prepareMask = async () => {
      try {
        const maskImage = new Image();
        maskImage.src = maskSource;
        await maskImage.decode();
        if (cancelled) return;

        const preparedMask = document.createElement("canvas");
        preparedMask.width = maskImage.naturalWidth;
        preparedMask.height = maskImage.naturalHeight;
        const maskContext = preparedMask.getContext("2d", {
          willReadFrequently: true,
        });
        if (!maskContext) throw new Error("Mask canvas is unavailable");
        maskContext.drawImage(maskImage, 0, 0);
        const maskName = maskSource.slice(maskSource.lastIndexOf("/") + 1);
        if (maskName === "Frankenstein.png") {
          removeEdgeColorBackground(
            maskContext,
            preparedMask.width,
            preparedMask.height,
          );
        } else if (MASKS_TO_REMOVE_BACKGROUND.has(maskName)) {
          removeEdgeBackground(
            maskContext,
            preparedMask.width,
            preparedMask.height,
          );
        }

        if (!cancelled) {
          preparedMaskRef.current = preparedMask;
          preparedMaskNameRef.current = maskName;
          preparedMaskScaleRef.current =
            maskName === "phantom.png"
              ? 1.5
              : maskName === "Frankenstein.png"
                ? 1.4
                : 1;
        }

        if (maskName === "zombie.png") {
          try {
            const eyeSheet = new Image();
            eyeSheet.src = `${import.meta.env.BASE_URL}masks/zombie-eye.png`;
            await eyeSheet.decode();
            if (cancelled) return;

            const createEyeSprite = (sourceX: number) => {
              const sprite = document.createElement("canvas");
              sprite.width = 224;
              sprite.height = 195;
              const spriteContext = sprite.getContext("2d", {
                willReadFrequently: true,
              });
              if (!spriteContext)
                throw new Error("Zombie eye canvas unavailable");
              spriteContext.drawImage(
                eyeSheet,
                sourceX,
                0,
                224,
                195,
                0,
                0,
                sprite.width,
                sprite.height,
              );
              removeEdgeBackground(spriteContext, sprite.width, sprite.height);
              return sprite;
            };

            zombieEyeSpritesRef.current = {
              lookLeft: [createEyeSprite(0), createEyeSprite(280)],
              lookRight: [createEyeSprite(510), createEyeSprite(800)],
            };
          } catch {
            zombieEyeSpritesRef.current = null;
          }

          try {
            const jawSheet = new Image();
            jawSheet.src = `${import.meta.env.BASE_URL}masks/zombie%20jaw.png`;
            await jawSheet.decode();
            if (cancelled) return;

            const createJawSprite = (sourceY: number) => {
              const sprite = document.createElement("canvas");
              sprite.width = 724;
              sprite.height = 212;
              const spriteContext = sprite.getContext("2d", {
                willReadFrequently: true,
              });
              if (!spriteContext)
                throw new Error("Zombie jaw canvas unavailable");
              spriteContext.drawImage(
                jawSheet,
                150,
                sourceY,
                724,
                212,
                0,
                0,
                sprite.width,
                sprite.height,
              );
              removeEdgeBackground(spriteContext, sprite.width, sprite.height);
              return sprite;
            };

            zombieJawSpritesRef.current = {
              closed: createJawSprite(300),
              open: createJawSprite(812),
            };
          } catch {
            zombieJawSpritesRef.current = null;
          }
        }

        if (maskName === "skull.png") {
          try {
            const eyeSheet = new Image();
            eyeSheet.src = `${import.meta.env.BASE_URL}masks/eye.png`;
            await eyeSheet.decode();
            if (cancelled) return;

            const createEyePair = (
              leftSourceX: number,
              rightSourceX: number,
              sourceY: number,
            ) => {
              const createEyeSprite = (sourceX: number) => {
                const sprite = document.createElement("canvas");
                sprite.width = 160;
                sprite.height = 165;
                const spriteContext = sprite.getContext("2d", {
                  willReadFrequently: true,
                });
                if (!spriteContext)
                  throw new Error("Skull eye canvas unavailable");
                spriteContext.drawImage(
                  eyeSheet,
                  sourceX,
                  sourceY,
                  sprite.width,
                  sprite.height,
                  0,
                  0,
                  sprite.width,
                  sprite.height,
                );
                removeCheckerboardBackground(
                  spriteContext,
                  sprite.width,
                  sprite.height,
                );
                return sprite;
              };

              return [
                createEyeSprite(leftSourceX),
                createEyeSprite(rightSourceX),
              ] as [HTMLCanvasElement, HTMLCanvasElement];
            };

            skullEyeSpritesRef.current = {
              lookLeft: createEyePair(15, 190, 330),
              lookCenter: createEyePair(695, 860, 110),
              lookRight: createEyePair(15, 190, 552),
              squintLeft: createEyePair(695, 860, 330),
              squintCenter: createEyePair(360, 525, 780),
              squintRight: createEyePair(695, 860, 552),
            };
          } catch {
            skullEyeSpritesRef.current = null;
          }

          try {
            const jawSheet = new Image();
            jawSheet.src = `${import.meta.env.BASE_URL}masks/jaw%20bone.png`;
            await jawSheet.decode();
            if (cancelled) return;

            const createJawSprite = (sourceX: number) => {
              const jawSprite = document.createElement("canvas");
              jawSprite.width = 512;
              jawSprite.height = 340;
              const jawContext = jawSprite.getContext("2d", {
                willReadFrequently: true,
              });
              if (!jawContext) throw new Error("Skull jaw canvas unavailable");
              jawContext.drawImage(
                jawSheet,
                sourceX,
                470,
                jawSprite.width,
                jawSprite.height,
                0,
                0,
                jawSprite.width,
                jawSprite.height,
              );
              removeCheckerboardBackground(
                jawContext,
                jawSprite.width,
                jawSprite.height,
              );
              return jawSprite;
            };

            skullJawSpritesRef.current = {
              closed: createJawSprite(0),
              open: createJawSprite(512),
            };
          } catch {
            skullJawSpritesRef.current = null;
          }
        }

        if (maskName === "phantom.png") {
          try {
            const eyeSheet = new Image();
            eyeSheet.src = `${import.meta.env.BASE_URL}masks/phantom-eye.png`;
            await eyeSheet.decode();
            if (cancelled) return;

            const createEyePair = (
              leftSourceX: number,
              rightSourceX: number,
              sourceY: number,
            ) => {
              const createEyeSprite = (sourceX: number) => {
                const sprite = document.createElement("canvas");
                sprite.width = 175;
                sprite.height = 165;
                const spriteContext = sprite.getContext("2d", {
                  willReadFrequently: true,
                });
                if (!spriteContext)
                  throw new Error("Phantom eye canvas unavailable");
                spriteContext.drawImage(
                  eyeSheet,
                  sourceX,
                  sourceY,
                  sprite.width,
                  sprite.height,
                  0,
                  0,
                  sprite.width,
                  sprite.height,
                );
                removeCheckerboardBackground(
                  spriteContext,
                  sprite.width,
                  sprite.height,
                );
                return sprite;
              };

              return [
                createEyeSprite(leftSourceX),
                createEyeSprite(rightSourceX),
              ] as [HTMLCanvasElement, HTMLCanvasElement];
            };

            phantomEyeSpritesRef.current = {
              lookLeft: createEyePair(45, 280, 205),
              lookCenter: createEyePair(45, 280, 0),
              lookRight: createEyePair(560, 800, 205),
              blink: createEyePair(560, 800, 710),
            };
          } catch {
            phantomEyeSpritesRef.current = null;
          }
        }
      } catch {
        if (!cancelled) {
          preparedMaskRef.current = null;
          preparedMaskNameRef.current = "";
        }
      }
    };

    void prepareMask();
    return () => {
      cancelled = true;
    };
  }, [maskSource]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!video || !canvas) return;

    const context = canvas.getContext("2d");
    if (!context) {
      onStatus("FACE TRACKING UNAVAILABLE");
      return;
    }

    let cancelled = false;
    let frameId = 0;
    let lastDetectionAt = -Infinity;
    let landmarker: FaceLandmarker | null = null;
    let tracks: FaceTrack[] = [];
    let nextTrackId = 1;
    let reportedStatus = "";
    let smoothedMouthOpening = 0;
    let smoothedJawOpening = 0;

    const reportStatus = (status: string) => {
      if (status !== reportedStatus) {
        reportedStatus = status;
        onStatus(status);
      }
    };

    const mapLandmark = (
      landmark: NormalizedLandmark,
      width: number,
      scale: number,
      offsetX: number,
      offsetY: number,
    ): Point => ({
      x: width - (landmark.x * video.videoWidth * scale + offsetX),
      y: landmark.y * video.videoHeight * scale + offsetY,
    });

    const findMaskPose = (
      landmarks: NormalizedLandmark[],
      width: number,
      height: number,
    ): MaskPose | null => {
      const scale = Math.max(
        width / video.videoWidth,
        height / video.videoHeight,
      );
      const offsetX = (width - video.videoWidth * scale) / 2;
      const offsetY = (height - video.videoHeight * scale) / 2;
      const point = (index: number) =>
        mapLandmark(landmarks[index], width, scale, offsetX, offsetY);
      const leftCheek = point(234);
      const rightCheek = point(454);
      const forehead = point(10);
      const chin = point(152);
      const firstEye = point(33);
      const secondEye = point(263);
      const [leftEye, rightEye] =
        firstEye.x < secondEye.x
          ? [firstEye, secondEye]
          : [secondEye, firstEye];
      const faceWidth = distance(leftCheek, rightCheek);
      const faceHeight = distance(forehead, chin);
      const maskScale = 1.3225;

      if (faceWidth < 24 || faceHeight < 24) return null;

      return {
        x: (leftCheek.x + rightCheek.x) / 2,
        y: (forehead.y + chin.y) / 2,
        width: faceWidth * 1.08 * maskScale,
        height: faceHeight * maskScale,
        angle: Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x),
      };
    };

    const findFaceFeatures = (
      landmarks: NormalizedLandmark[],
      width: number,
      height: number,
      jawOpening: number,
      mouthOpening: number,
      mouthRoundness: number,
      mouthStretch: number,
    ): FaceFeatures => {
      const scale = Math.max(
        width / video.videoWidth,
        height / video.videoHeight,
      );
      const offsetX = (width - video.videoWidth * scale) / 2;
      const offsetY = (height - video.videoHeight * scale) / 2;
      const point = (index: number) =>
        mapLandmark(landmarks[index], width, scale, offsetX, offsetY);
      const eye = (
        outerIndex: number,
        innerIndex: number,
        upperIndex: number,
        lowerIndex: number,
        irisIndex: number,
      ): EyeOpening => {
        const outer = point(outerIndex);
        const inner = point(innerIndex);
        const center = {
          x: (outer.x + inner.x) / 2,
          y: (outer.y + inner.y) / 2,
        };
        return {
          center,
          width: distance(outer, inner),
          height: distance(point(upperIndex), point(lowerIndex)),
          iris: landmarks.length > irisIndex ? point(irisIndex) : center,
        };
      };
      const leftCheek = point(234);
      const rightCheek = point(454);
      const forehead = point(10);
      const chin = point(152);
      const leftMouthCorner = point(61);
      const rightMouthCorner = point(291);
      const upperLip = point(13);
      const lowerLip = point(14);
      const eyes = [
        eye(33, 133, 159, 145, 468),
        eye(263, 362, 386, 374, 473),
      ] as [EyeOpening, EyeOpening];
      if (eyes[0].center.x > eyes[1].center.x) eyes.reverse();

      return {
        faceWidth: distance(leftCheek, rightCheek),
        faceHeight: distance(forehead, chin),
        eyes,
        mouth: {
          center: {
            x: (upperLip.x + lowerLip.x) / 2,
            y: (upperLip.y + lowerLip.y) / 2,
          },
          width: distance(leftMouthCorner, rightMouthCorner),
          gap: distance(upperLip, lowerLip),
        },
        jawOpening,
        mouthOpening,
        mouthRoundness,
        mouthStretch,
      };
    };

    const updateTracks = (detectedPoses: MaskPose[], now: number) => {
      const activeTracks = tracks.filter(
        (track) => now - track.lastSeenAt <= 220,
      );
      const possibleMatches: Array<{
        trackIndex: number;
        poseIndex: number;
        cost: number;
      }> = [];

      activeTracks.forEach((track, trackIndex) => {
        detectedPoses.forEach((pose, poseIndex) => {
          const cost =
            distance(track.pose, pose) / Math.max(track.pose.width, pose.width);
          if (cost <= 1.15)
            possibleMatches.push({ trackIndex, poseIndex, cost });
        });
      });
      possibleMatches.sort((first, second) => first.cost - second.cost);

      const ambiguousTracks = new Set<number>();
      const ambiguousPoses = new Set<number>();
      activeTracks.forEach((_track, trackIndex) => {
        const matches = possibleMatches.filter(
          (match) => match.trackIndex === trackIndex,
        );
        if (matches.length > 1 && matches[1].cost - matches[0].cost < 0.18) {
          ambiguousTracks.add(trackIndex);
          ambiguousPoses.add(matches[0].poseIndex);
          ambiguousPoses.add(matches[1].poseIndex);
        }
      });
      detectedPoses.forEach((_pose, poseIndex) => {
        const matches = possibleMatches.filter(
          (match) => match.poseIndex === poseIndex,
        );
        if (matches.length > 1 && matches[1].cost - matches[0].cost < 0.18) {
          ambiguousPoses.add(poseIndex);
          ambiguousTracks.add(matches[0].trackIndex);
          ambiguousTracks.add(matches[1].trackIndex);
        }
      });

      const assignedTracks = new Set<number>();
      const assignedPoses = new Map<number, number>();
      possibleMatches.forEach((match) => {
        if (
          ambiguousTracks.has(match.trackIndex) ||
          ambiguousPoses.has(match.poseIndex) ||
          assignedTracks.has(match.trackIndex) ||
          assignedPoses.has(match.poseIndex)
        ) {
          return;
        }
        assignedTracks.add(match.trackIndex);
        assignedPoses.set(match.poseIndex, match.trackIndex);
      });

      const updatedTracks: FaceTrack[] = activeTracks.flatMap(
        (track, trackIndex) => {
          const poseIndex = [...assignedPoses.entries()].find(
            ([, assignedTrackIndex]) => assignedTrackIndex === trackIndex,
          )?.[0];
          if (poseIndex === undefined) {
            return now - track.lastSeenAt <= 160 ? [track] : [];
          }

          const target = detectedPoses[poseIndex];
          return [
            {
              ...track,
              lastSeenAt: now,
              pose: {
                x: track.pose.x + (target.x - track.pose.x) * 0.38,
                y: track.pose.y + (target.y - track.pose.y) * 0.38,
                width:
                  track.pose.width + (target.width - track.pose.width) * 0.38,
                height:
                  track.pose.height +
                  (target.height - track.pose.height) * 0.38,
                angle: interpolateAngle(track.pose.angle, target.angle, 0.38),
              },
            },
          ];
        },
      );

      detectedPoses.forEach((pose, poseIndex) => {
        if (assignedPoses.has(poseIndex) || ambiguousPoses.has(poseIndex))
          return;
        if (
          possibleMatches.some((match) => match.poseIndex === poseIndex) ||
          updatedTracks.length >= MAX_TRACKED_FACES
        ) {
          return;
        }

        updatedTracks.push({ id: nextTrackId++, pose, lastSeenAt: now });
      });

      tracks = updatedTracks;
      return tracks.length;
    };

    const drawMask = (pose: MaskPose) => {
      const preparedMask = preparedMaskRef.current;
      if (!preparedMask) return;

      const width = pose.width * preparedMaskScaleRef.current;
      const height = pose.height * preparedMaskScaleRef.current;
      context.save();
      context.translate(pose.x, pose.y);
      context.rotate(pose.angle);
      context.drawImage(
        preparedMask,
        100,
        35,
        824,
        950,
        -width / 2,
        -height / 2,
        width,
        height,
      );
      context.restore();

      const features = faceFeaturesRef.current;
      if (!features || features.faceWidth === 0 || features.faceHeight === 0)
        return;

      if (preparedMaskNameRef.current === "Frankenstein.png") {
        context.save();
        context.translate(pose.x, pose.y);
        context.rotate(pose.angle);

        features.eyes.forEach((eye, index) => {
          const anchor = FRANKENSTEIN_EYE_ANCHORS[index];
          const eyeX = (anchor.x - 0.5) * width;
          const eyeY = (anchor.y - 0.5) * height;
          const gazeX = Math.max(
            -1,
            Math.min(1, (eye.iris.x - eye.center.x) / (eye.width * 0.38)),
          );
          const gazeY = Math.max(
            -1,
            Math.min(1, (eye.iris.y - eye.center.y) / (eye.height * 0.6)),
          );
          const blink = Math.max(
            0.18,
            Math.min(1, eye.height / (eye.width * 0.3)),
          );
          const pupilX = eyeX + gazeX * width * 0.024;
          const pupilY = eyeY + gazeY * height * 0.018;

          context.fillStyle = "#f5f2df";
          context.beginPath();
          context.ellipse(
            eyeX,
            eyeY,
            width * 0.047,
            height * 0.033 * blink,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.fillStyle = "#241923";
          context.beginPath();
          context.ellipse(
            pupilX,
            pupilY,
            width * 0.016,
            height * 0.018 * blink,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.fillStyle = "#ffffff";
          context.beginPath();
          context.arc(
            pupilX - width * 0.005,
            pupilY - height * 0.006,
            width * 0.004,
            0,
            Math.PI * 2,
          );
          context.fill();
        });

        const mouthOpen = features.mouthOpening;
        if (mouthOpen > 0) {
          const mouthX = (FRANKENSTEIN_MOUTH_ANCHOR.x - 0.5) * width;
          const mouthY = (FRANKENSTEIN_MOUTH_ANCHOR.y - 0.5) * height;
          const mouthRadiusX =
            width *
            (0.19 +
              features.mouthStretch * 0.018 -
              features.mouthRoundness * 0.025);
          const mouthRadiusY =
            height *
            (0.006 + mouthOpen * 0.075 + features.mouthRoundness * 0.018);

          context.fillStyle = "#160d13";
          context.beginPath();
          context.ellipse(
            mouthX,
            mouthY,
            mouthRadiusX,
            mouthRadiusY,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();

          context.fillStyle = "#f1e5ca";
          for (let tooth = 0; tooth < 5; tooth += 1) {
            const toothX =
              mouthX - mouthRadiusX * 0.8 + tooth * mouthRadiusX * 0.4;
            context.fillRect(
              toothX,
              mouthY - mouthRadiusY * 0.72,
              mouthRadiusX * 0.18,
              mouthRadiusY * 0.42,
            );
          }
          context.fillStyle = "#8c3040";
          context.beginPath();
          context.ellipse(
            mouthX,
            mouthY + mouthRadiusY * 0.42,
            mouthRadiusX * 0.38,
            mouthRadiusY * 0.26,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
        }

        context.restore();
        return;
      }

      const anchors = MASK_FEATURE_ANCHORS[preparedMaskNameRef.current];
      if (!anchors) return;

      context.save();
      context.translate(pose.x, pose.y);
      context.rotate(pose.angle);
      const mouthOpen = features.mouthOpening;
      if (anchors.animateFeatures) {
        features.eyes.forEach((eye, index) => {
          const anchor = anchors.eyes[index];
          const eyeX = (anchor.x - 0.5) * width;
          const eyeY = (anchor.y - 0.5) * height;
          const gazeX = Math.max(
            -1,
            Math.min(1, (eye.iris.x - eye.center.x) / (eye.width * 0.38)),
          );
          const gazeY = Math.max(
            -1,
            Math.min(1, (eye.iris.y - eye.center.y) / (eye.height * 0.6)),
          );
          const blink = Math.max(
            0.18,
            Math.min(1, eye.height / (eye.width * 0.3)),
          );
          const pupilX = eyeX + gazeX * width * 0.024;
          const pupilY = eyeY + gazeY * height * 0.018;
          const eyeRadiusX = width * anchors.eyeRadiusX;
          const eyeRadiusY = height * anchors.eyeRadiusY;

          if (preparedMaskNameRef.current === "skull.png") {
            const skullEyeSprites = skullEyeSpritesRef.current;
            if (skullEyeSprites) {
              const eyeState =
                blink < 0.72
                  ? gazeX < -0.2
                    ? skullEyeSprites.squintLeft
                    : gazeX > 0.2
                      ? skullEyeSprites.squintRight
                      : skullEyeSprites.squintCenter
                  : gazeX < -0.2
                    ? skullEyeSprites.lookLeft
                    : gazeX > 0.2
                      ? skullEyeSprites.lookRight
                      : skullEyeSprites.lookCenter;
              const eyeSprite = eyeState[index];
              const eyeWidth = eyeRadiusX * 2;
              const eyeHeight = eyeRadiusY * 2 * Math.max(0.65, blink);
              context.drawImage(
                eyeSprite,
                eyeX - eyeWidth / 2,
                eyeY - eyeHeight / 2,
                eyeWidth,
                eyeHeight,
              );
              return;
            }
          }

          if (preparedMaskNameRef.current === "phantom.png") {
            const phantomEyeSprites = phantomEyeSpritesRef.current;
            if (phantomEyeSprites) {
              const eyeState =
                blink < 0.72
                  ? phantomEyeSprites.blink
                  : gazeX < -0.2
                    ? phantomEyeSprites.lookLeft
                    : gazeX > 0.2
                      ? phantomEyeSprites.lookRight
                      : phantomEyeSprites.lookCenter;
              const eyeWidth = eyeRadiusX * 2.15;
              const eyeHeight = eyeRadiusY * 2.15 * Math.max(0.72, blink);
              context.drawImage(
                eyeState[index],
                eyeX - eyeWidth / 2,
                eyeY - eyeHeight / 2,
                eyeWidth,
                eyeHeight,
              );
              return;
            }
          }

          if (
            preparedMaskNameRef.current === "pumpkin.png" ||
            preparedMaskNameRef.current === "jack.png"
          ) {
            const pupilX = eyeX + gazeX * width * 0.022;
            const pupilY = eyeY + gazeY * height * 0.018;
            const pupilRadiusX = width * 0.022;
            const pupilRadiusY = height * 0.023 * Math.max(0.72, blink);
            const glow = context.createRadialGradient(
              pupilX,
              pupilY,
              pupilRadiusX * 0.12,
              pupilX,
              pupilY,
              pupilRadiusX * 2.6,
            );
            glow.addColorStop(0, "rgba(255, 255, 205, 0.95)");
            glow.addColorStop(0.28, "rgba(255, 219, 75, 0.9)");
            glow.addColorStop(1, "rgba(255, 111, 15, 0)");

            context.save();
            context.shadowColor = "#ffbd35";
            context.shadowBlur = width * 0.018;
            context.fillStyle = glow;
            context.beginPath();
            context.ellipse(
              pupilX,
              pupilY,
              pupilRadiusX * 2.6,
              pupilRadiusY * 2.6,
              0,
              0,
              Math.PI * 2,
            );
            context.fill();
            context.shadowBlur = 0;
            context.fillStyle = "#fff8c8";
            context.beginPath();
            context.ellipse(
              pupilX,
              pupilY,
              pupilRadiusX,
              pupilRadiusY,
              0,
              0,
              Math.PI * 2,
            );
            context.fill();
            context.restore();
            return;
          }

          if (preparedMaskNameRef.current === "zombie.png") {
            const zombieEyeSprites = zombieEyeSpritesRef.current;
            if (zombieEyeSprites) {
              const eyeSprites =
                gazeX < -0.2
                  ? zombieEyeSprites.lookLeft
                  : zombieEyeSprites.lookRight;
              const eyeWidth = eyeRadiusX * 2.15;
              const eyeHeight = eyeRadiusY * 2.1 * blink;
              context.drawImage(
                eyeSprites[index],
                eyeX - eyeWidth / 2,
                eyeY - eyeHeight / 2,
                eyeWidth,
                eyeHeight,
              );
              return;
            }
          }

          context.fillStyle = "#e8dfd2";
          context.beginPath();
          context.ellipse(
            eyeX,
            eyeY,
            eyeRadiusX * 0.88,
            eyeRadiusY * 0.9 * blink,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.fillStyle = "#251b1d";
          context.beginPath();
          context.ellipse(
            pupilX,
            pupilY,
            eyeRadiusX * 0.4,
            eyeRadiusY * 0.72 * blink,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.fillStyle = "#ffffff";
          context.beginPath();
          context.arc(
            pupilX - eyeRadiusX * 0.12,
            pupilY - eyeRadiusY * 0.18,
            eyeRadiusX * 0.1,
            0,
            Math.PI * 2,
          );
          context.fill();
        });

        const zombieJawSprites =
          preparedMaskNameRef.current === "zombie.png"
            ? zombieJawSpritesRef.current
            : null;
        const skullJawSprites =
          preparedMaskNameRef.current === "skull.png"
            ? skullJawSpritesRef.current
            : null;
        if (skullJawSprites) {
          const jawProgress = features.jawOpening;
          const jawMotion = Math.max(0, (jawProgress - 0.08) / 0.92);
          const jawWidth = width * 0.58;
          const jawHeight = height * 0.28;
          const jawY = height * (0.2 + jawMotion * 0.09);

          context.save();
          context.globalCompositeOperation = "destination-out";
          context.beginPath();
          context.ellipse(
            0,
            height * 0.3,
            width * 0.3,
            height * 0.15,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.globalCompositeOperation = "source-over";
          context.translate(0, jawY);
          context.rotate(jawMotion * 0.18);
          context.drawImage(
            jawProgress < 0.28 ? skullJawSprites.closed : skullJawSprites.open,
            -jawWidth / 2,
            0,
            jawWidth,
            jawHeight,
          );
          context.restore();
        } else if (zombieJawSprites) {
          const jawProgress = features.jawOpening;
          const jawWidth = width * 0.82;
          const jawHeight = height * 0.22;
          const jawY = height * (0.24 + jawProgress * 0.035);

          context.save();
          context.translate(0, jawY);
          context.rotate(jawProgress * 0.16);
          context.drawImage(
            jawProgress < 0.3 ? zombieJawSprites.closed : zombieJawSprites.open,
            -jawWidth / 2,
            0,
            jawWidth,
            jawHeight,
          );
          context.restore();
        } else if (anchors.animateMouth !== false && mouthOpen > 0.02) {
          const mouthX = (anchors.mouth.x - 0.5) * width;
          const mouthY = (anchors.mouth.y - 0.5) * height;
          const mouthRadiusX =
            width * anchors.mouthRadiusX * (1 + features.mouthStretch * 0.06);
          const mouthRadiusY =
            height *
            anchors.mouthRadiusY *
            (0.12 + mouthOpen * 1.2 + features.mouthRoundness * 0.18);

          context.fillStyle = "#261116";
          context.beginPath();
          context.ellipse(
            mouthX,
            mouthY,
            mouthRadiusX,
            mouthRadiusY,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
          context.fillStyle = "#eee2d1";
          for (let tooth = 0; tooth < 5; tooth += 1) {
            const toothX =
              mouthX - mouthRadiusX * 0.78 + tooth * mouthRadiusX * 0.39;
            context.fillRect(
              toothX,
              mouthY - mouthRadiusY * 0.78,
              mouthRadiusX * 0.16,
              mouthRadiusY * 0.28,
            );
          }
          context.fillStyle = "#873749";
          context.beginPath();
          context.ellipse(
            mouthX,
            mouthY + mouthRadiusY * 0.48,
            mouthRadiusX * 0.32,
            mouthRadiusY * 0.2,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
        }
      } else {
        context.globalCompositeOperation = "destination-out";
        anchors.eyes.forEach((eye) => {
          context.beginPath();
          context.ellipse(
            (eye.x - 0.5) * width,
            (eye.y - 0.5) * height,
            width * anchors.eyeRadiusX,
            height * anchors.eyeRadiusY,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
        });

        if (mouthOpen > 0) {
          context.beginPath();
          context.ellipse(
            (anchors.mouth.x - 0.5) * width,
            (anchors.mouth.y - 0.5) * height,
            width * anchors.mouthRadiusX,
            height * anchors.mouthRadiusY * mouthOpen,
            0,
            0,
            Math.PI * 2,
          );
          context.fill();
        }
      }
      context.restore();
    };

    const drawFrame = (now: number) => {
      if (cancelled) return;
      frameId = window.requestAnimationFrame(drawFrame);

      const bounds = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const backingWidth = Math.round(bounds.width * pixelRatio);
      const backingHeight = Math.round(bounds.height * pixelRatio);

      if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
        canvas.width = backingWidth;
        canvas.height = backingHeight;
      }

      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      if (
        !landmarker ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0
      ) {
        return;
      }

      if (now - lastDetectionAt >= 40) {
        lastDetectionAt = now;

        try {
          const result = landmarker.detectForVideo(video, now);
          const landmarks = result.faceLandmarks[0];
          const categories = result.faceBlendshapes[0]?.categories ?? [];
          const blendshape = (name: string) =>
            categories.find((category) => category.categoryName === name)
              ?.score ?? 0;
          const jawOpen = blendshape("jawOpen");
          const mouthRoundness = Math.max(
            blendshape("mouthFunnel"),
            blendshape("mouthPucker"),
          );
          const mouthStretch = Math.max(
            blendshape("mouthStretchLeft"),
            blendshape("mouthStretchRight"),
          );
          const lipMovement = Math.max(
            blendshape("mouthLowerDownLeft"),
            blendshape("mouthLowerDownRight"),
          );
          const mouthActivity = Math.max(
            jawOpen,
            lipMovement * 0.82,
            blendshape("mouthFunnel") * 0.72,
            blendshape("mouthPucker") * 0.58,
            mouthStretch * 0.42,
          );
          const targetMouthOpening = Math.max(
            0,
            Math.min(1, (mouthActivity - 0.06) / 0.62),
          );
          const targetJawOpening = Math.max(
            0,
            Math.min(1, (jawOpen - 0.06) / 0.54),
          );
          smoothedMouthOpening +=
            (targetMouthOpening - smoothedMouthOpening) *
            (targetMouthOpening > smoothedMouthOpening ? 0.62 : 0.34);
          smoothedJawOpening +=
            (targetJawOpening - smoothedJawOpening) *
            (targetJawOpening > smoothedJawOpening ? 0.62 : 0.34);
          faceFeaturesRef.current = landmarks
            ? findFaceFeatures(
                landmarks,
                bounds.width,
                bounds.height,
                smoothedJawOpening,
                smoothedMouthOpening,
                mouthRoundness,
                mouthStretch,
              )
            : null;
          const detectedPoses = result.faceLandmarks
            .slice(0, MAX_TRACKED_FACES)
            .map((landmarks) =>
              findMaskPose(landmarks, bounds.width, bounds.height),
            )
            .filter((pose): pose is MaskPose => pose !== null);
          const count = updateTracks(detectedPoses, now);
          reportStatus(
            count === 0
              ? "SEARCHING FOR FACE"
              : `${count} ${count === 1 ? "FACE" : "FACES"} TRACKED`,
          );
        } catch {
          tracks = [];
          smoothedMouthOpening = 0;
          smoothedJawOpening = 0;
          faceFeaturesRef.current = null;
          reportStatus("FACE TRACKING UNAVAILABLE");
        }
      }

      tracks.forEach((track) => drawMask(track.pose));
    };

    const initialize = async () => {
      reportStatus("LOADING LOCAL FACE MODEL");

      try {
        const fileset = await FilesetResolver.forVisionTasks(
          `${import.meta.env.BASE_URL}mediapipe/wasm`,
        );
        if (cancelled) return;

        try {
          landmarker = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: MAX_TRACKED_FACES,
            minFaceDetectionConfidence: 0.55,
            minFacePresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: false,
          });
        } catch {
          if (cancelled) return;

          landmarker = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numFaces: MAX_TRACKED_FACES,
            minFaceDetectionConfidence: 0.55,
            minFacePresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: false,
          });
        }

        if (cancelled) {
          landmarker.close();
          landmarker = null;
          return;
        }

        reportStatus("SEARCHING FOR FACE");
        frameId = window.requestAnimationFrame(drawFrame);
      } catch {
        if (!cancelled) reportStatus("FACE TRACKING UNAVAILABLE");
      }
    };

    void initialize();

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frameId);
      landmarker?.close();
      context.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [onStatus, videoRef]);

  return (
    <canvas ref={canvasRef} className="tracking-overlay" aria-hidden="true" />
  );
}
