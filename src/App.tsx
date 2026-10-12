import { useCallback, useEffect, useRef, useState } from "react";
import { CameraBackground } from "./CameraBackground";
import { FaceMaskOverlay } from "./FaceMaskOverlay";
import "./App.css";

type CameraStatus = "off" | "requesting" | "live";

const scenes = [
  {
    file: "forest.png",
    name: "The Veilwood",
    detail: "A red moon over the whispering trees.",
  },
  {
    file: "graveyard.png",
    name: "Blackthorn Cemetery",
    detail: "The gates stand open tonight.",
  },
  {
    file: "house.png",
    name: "The Looking House",
    detail: "Something waits behind the glass.",
  },
];

const masks = [
  { file: "Frankenstein.png", name: "Frankenstein" },
  { file: "doll.png", name: "Doll" },
  { file: "doll2.png", name: "Doll II" },
  { file: "jack.png", name: "Jack-o'-lantern" },
  { file: "phantom.png", name: "Phantom" },
  { file: "pumpkin.png", name: "Pumpkin" },
  { file: "skull.png", name: "Skull" },
  { file: "specter.svg", name: "Specter" },
  { file: "thorn.svg", name: "Thorn" },
  { file: "zombie.png", name: "Zombie" },
];

function getNextMask(currentMask: string) {
  const currentIndex = masks.findIndex((mask) => mask.file === currentMask);
  return masks[(currentIndex + 1) % masks.length].file;
}

function cameraErrorMessage(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Camera permission was denied. The slideshow is still available.";
    }
    if (error.name === "NotFoundError") {
      return "No camera was found. The slideshow is still available.";
    }
    if (error.name === "NotReadableError") {
      return "The camera is unavailable or in use. The slideshow is still available.";
    }
  }

  return "Could not start the camera. The slideshow is still available.";
}

function App() {
  const [activeScene, setActiveScene] = useState(0);
  const [intervalSeconds, setIntervalSeconds] = useState(10);
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedMask, setSelectedMask] = useState("specter.svg");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("off");
  const [backgroundReady, setBackgroundReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [trackingStatus, setTrackingStatus] = useState(
    "LOADING LOCAL FACE MODEL",
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraRequestId = useRef(0);

  const stopCamera = useCallback((error: string | null = null) => {
    cameraRequestId.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraStatus("off");
    setBackgroundReady(false);
    setCameraError(error);
  }, []);

  useEffect(() => {
    if (cameraStatus === "requesting") return;
    if (cameraStatus !== "live" && !isPlaying) return;

    const timer = window.setInterval(() => {
      setActiveScene((current) => {
        if (cameraStatus !== "live") return (current + 1) % scenes.length;

        const next = Math.floor(Math.random() * (scenes.length - 1));
        return next >= current ? next + 1 : next;
      });
    }, intervalSeconds * 1000);

    return () => window.clearInterval(timer);
  }, [cameraStatus, intervalSeconds, isPlaying]);

  useEffect(() => {
    const video = videoRef.current;
    const releaseWhenHidden = () => {
      if (document.visibilityState !== "visible") stopCamera();
    };
    const releaseOnPageHide = () => stopCamera();

    document.addEventListener("visibilitychange", releaseWhenHidden);
    window.addEventListener("pagehide", releaseOnPageHide);

    return () => {
      document.removeEventListener("visibilitychange", releaseWhenHidden);
      window.removeEventListener("pagehide", releaseOnPageHide);
      cameraRequestId.current += 1;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (video) {
        video.pause();
        video.srcObject = null;
      }
    };
  }, [stopCamera]);

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera access is unavailable. The slideshow is still available.",
      );
      return;
    }

    const requestId = ++cameraRequestId.current;
    setCameraError(null);
    setBackgroundReady(false);
    setCameraStatus("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user" },
      });

      if (requestId !== cameraRequestId.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      if (document.visibilityState !== "visible") {
        stream.getTracks().forEach((track) => track.stop());
        stopCamera();
        return;
      }

      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        stopCamera(
          "The camera preview is unavailable. The slideshow is still available.",
        );
        return;
      }

      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();

      if (requestId !== cameraRequestId.current) return;

      setSelectedMask((currentMask) => getNextMask(currentMask));
      setCameraStatus("live");
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener(
          "ended",
          () => {
            if (streamRef.current === stream) {
              stopCamera(
                "Camera disconnected. The slideshow is still available.",
              );
            }
          },
          { once: true },
        );
      });
    } catch (error) {
      if (requestId === cameraRequestId.current) {
        stopCamera(cameraErrorMessage(error));
      }
    }
  };

  const showScene = (offset: number) => {
    setActiveScene(
      (current) => (current + offset + scenes.length) % scenes.length,
    );
  };

  const toggleFullscreen = async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }

    await document.documentElement.requestFullscreen();
  };

  const scene = scenes[activeScene];
  const cameraIsLive = cameraStatus === "live";

  return (
    <main className={`mirror-stage${cameraIsLive ? " camera-mode" : ""}`}>
      <video
        ref={videoRef}
        className={`camera-feed${cameraIsLive ? " is-visible" : ""}${backgroundReady ? " is-segmented" : ""}`}
        autoPlay
        playsInline
        muted
        aria-label="Mirrored live camera preview"
        aria-hidden={!cameraIsLive}
      />
      {cameraIsLive && (
        <CameraBackground
          videoRef={videoRef}
          backgroundSource={`${import.meta.env.BASE_URL}backgrounds/${scene.file}`}
          onReadyChange={setBackgroundReady}
        />
      )}
      {cameraIsLive && (
        <FaceMaskOverlay
          videoRef={videoRef}
          onStatus={setTrackingStatus}
          maskSource={`${import.meta.env.BASE_URL}masks/${selectedMask}`}
        />
      )}
      {!cameraIsLive && (
        <img
          key={scene.file}
          className="scene-image"
          src={`${import.meta.env.BASE_URL}backgrounds/${scene.file}`}
          alt=""
        />
      )}
      <div className="scene-shade" aria-hidden="true" />

      <header className="topbar">
        <div className="mode-status" role="status" aria-live="polite">
          <span className="status-light" />
          <span>
            {cameraIsLive
              ? "CAMERA ON"
              : cameraStatus === "requesting"
                ? "STARTING CAMERA"
                : "CAMERA OFF"}
          </span>
          <span className="status-divider">/</span>
          <span className="mode-name">
            {cameraIsLive ? "LIVE PREVIEW" : "SLIDESHOW"}
          </span>
        </div>
      </header>

      {cameraError && (
        <p className="camera-error" role="alert">
          {cameraError}
        </p>
      )}

      {!cameraIsLive && (
        <section className="scene-caption" aria-live="polite">
          <p className="scene-index">
            SCENE {String(activeScene + 1).padStart(2, "0")} <span>/</span>{" "}
            {String(scenes.length).padStart(2, "0")}
          </p>
          <h1>{scene.name}</h1>
          <p className="scene-detail">{scene.detail}</p>
        </section>
      )}

      <footer className="operator-bar">
        {cameraIsLive ? (
          <>
            <div className="live-preview-label">
              LIVE PREVIEW <span>{trackingStatus}</span>
            </div>
            <label className="mask-picker">
              <span>MASK</span>
              <select
                value={selectedMask}
                onChange={(event) => setSelectedMask(event.target.value)}
                aria-label="Choose face mask"
              >
                {masks.map((mask) => (
                  <option key={mask.file} value={mask.file}>
                    {mask.name}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : (
          <div className="rotation-control">
            <label htmlFor="rotation-interval">ROTATION INTERVAL</label>
            <div className="range-row">
              <input
                id="rotation-interval"
                type="range"
                min="3"
                max="60"
                step="1"
                value={intervalSeconds}
                onChange={(event) =>
                  setIntervalSeconds(Number(event.target.value))
                }
              />
              <output htmlFor="rotation-interval">{intervalSeconds}s</output>
            </div>
          </div>
        )}

        {!cameraIsLive && (
          <div className="scene-controls" aria-label="Slideshow controls">
            <button
              type="button"
              onClick={() => showScene(-1)}
              aria-label="Previous scene"
            >
              PREV
            </button>
            <button
              type="button"
              className="play-control"
              onClick={() => setIsPlaying((playing) => !playing)}
              aria-label={isPlaying ? "Pause slideshow" : "Resume slideshow"}
            >
              {isPlaying ? "PAUSE" : "RESUME"}
            </button>
            <button
              type="button"
              onClick={() => showScene(1)}
              aria-label="Next scene"
            >
              NEXT
            </button>
          </div>
        )}

        <div className="operator-actions">
          <button
            type="button"
            className="camera-control"
            onClick={() => {
              if (cameraStatus === "off") {
                void startCamera();
              } else {
                stopCamera();
              }
            }}
            aria-label={
              cameraIsLive
                ? "Turn camera off"
                : cameraStatus === "requesting"
                  ? "Cancel camera request"
                  : "Turn camera on"
            }
          >
            {cameraIsLive
              ? "CAMERA OFF"
              : cameraStatus === "requesting"
                ? "CANCEL"
                : "CAMERA ON"}
          </button>
          <button
            type="button"
            className="fullscreen-control"
            onClick={toggleFullscreen}
          >
            FULLSCREEN
          </button>
        </div>
      </footer>
    </main>
  );
}

export default App;
