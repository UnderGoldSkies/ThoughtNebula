import {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
  Suspense,
  type FC,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, useGLTF } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { UMAP } from "umap-js";
import * as THREE from "three";
import { cos_sim } from "@huggingface/transformers";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { DEFAULT_SENTENCES } from "./constants";
import Logo from "./components/Logo";
import BackgroundMusic from "./components/BackgroundMusic";
import { useModel } from "./components/useModel";

type QualityLevel = "low" | "medium" | "high";

interface QualityPreset {
  label: string;
  dpr: [number, number];
  antialias: boolean;
  enableBloom: boolean;
  bloomIntensity: number;
  bloomHeight: number;
  powerPreference: WebGLPowerPreference;
  menu: {
    pointCount: number;
    starCount: number;
    starFactor: number;
  };
  scene: {
    starCount: number;
    starFactor: number;
  };
}

const QUALITY_PRESETS: Record<QualityLevel, QualityPreset> = {
  low: {
    label: "Battery saver",
    dpr: [0.65, 0.85],
    antialias: false,
    enableBloom: false,
    bloomIntensity: 0.8,
    bloomHeight: 240,
    powerPreference: "default",
    menu: { pointCount: 4000, starCount: 800, starFactor: 2 },
    scene: { starCount: 600, starFactor: 2 },
  },
  medium: {
    label: "Balanced",
    dpr: [1, 1.5],
    antialias: true,
    enableBloom: true,
    bloomIntensity: 1.1,
    bloomHeight: 260,
    powerPreference: "default",
    menu: { pointCount: 15000, starCount: 3000, starFactor: 6 },
    scene: { starCount: 2500, starFactor: 5 },
  },
  high: {
    label: "Max fidelity",
    dpr: [1, 2],
    antialias: true,
    enableBloom: true,
    bloomIntensity: 1.5,
    bloomHeight: 320,
    powerPreference: "high-performance",
    menu: { pointCount: 25000, starCount: 5000, starFactor: 8 },
    scene: { starCount: 4000, starFactor: 7 },
  },
};

const detectQualityPreset = (): QualityLevel => {
  if (typeof navigator === "undefined") return "high";
  const isMobile =
    /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) ||
    (typeof window !== "undefined" &&
      window.matchMedia("(max-width: 768px)").matches);
  const lowMemory = (navigator as Navigator & { deviceMemory?: number })
    .deviceMemory
    ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory! <= 4
    : false;
  const fewCores =
    typeof navigator.hardwareConcurrency === "number" &&
    navigator.hardwareConcurrency <= 4;
  if (isMobile) return "low";
  if (lowMemory || fewCores) return "medium";
  return "high";
};

const warpToBrainShape = (p: THREE.Vector3): THREE.Vector3 => {
  const v = p.clone();
  v.y *= 1.4;
  v.z *= 0.7;

  const signX = Math.sign(v.x) || (Math.random() < 0.5 ? -1 : 1);
  const lobeOffset = 0.5;
  const depthFactor = Math.pow(Math.abs(v.z), 0.3);
  v.x = v.x * 0.6 + signX * lobeOffset * depthFactor;

  const rXZ = Math.sqrt(v.x * v.x + v.z * v.z);
  const bulge = 1 + 0.3 * Math.exp(-rXZ * rXZ * 0.5);
  v.y *= bulge;

  return v;
};

const MainMenuUI: FC<{ onLoadModel: () => void }> = ({ onLoadModel }) => (
  <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none px-4">
    <div className="max-w-5xl w-full bg-white/85 border border-gray-200 backdrop-blur-xl rounded-2xl shadow-2xl p-8 sm:p-10 pointer-events-auto text-gray-900">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        <div className="space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full bg-gray-900 text-white px-3 py-1 text-sm">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Web-native embeddings · 3D semantic brain
          </div>
          <div className="flex items-center gap-3">
            <Logo className="w-12 h-12 rounded-xl shadow-md" />
            <h1 className="text-4xl sm:text-5xl font-bold leading-tight">
              ThoughtNebula
            </h1>
          </div>
          <p className="text-base sm:text-lg text-gray-700">
            Load the model in-browser and explore your text as neuron sparks inside a brain-shaped volume. No servers, no uploads.
          </p>
          <div className="flex flex-wrap gap-3 text-sm text-gray-800">
            <span className="px-3 py-1 rounded-full bg-gray-100 border border-gray-200">
              EmbeddingGemma · Transformers.js
            </span>
            <span className="px-3 py-1 rounded-full bg-gray-100 border border-gray-200">
              UMAP 3D projection
            </span>
            <span className="px-3 py-1 rounded-full bg-gray-100 border border-gray-200">
              Runs entirely local
            </span>
          </div>
          <div className="flex gap-3 flex-wrap">
            <button
              onClick={onLoadModel}
              className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-3 rounded-xl shadow-lg shadow-blue-900/20 transition-all duration-200"
            >
              Load Model & Demo
            </button>
            <div className="flex items-center text-sm text-gray-800">
              <div className="h-10 w-10 rounded-lg bg-gray-900 text-white flex items-center justify-center font-semibold mr-3">
                40MB
              </div>
              <div>
                <div className="font-semibold">Download once</div>
                <div className="text-gray-600">Cached in your browser</div>
              </div>
            </div>
          </div>
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 text-sm text-gray-800">
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
              <div className="text-gray-500">How it works</div>
              <ul className="list-disc list-inside leading-relaxed text-gray-800">
                <li>Embed your lines of text</li>
                <li>UMAP compresses to 3D</li>
                <li>Points light up by similarity</li>
              </ul>
            </div>
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
              <div className="text-gray-500">Best on</div>
              <div className="font-semibold text-gray-900">Desktop WebGPU / WASM</div>
              <div className="text-gray-600">Mobile works; lower quality preset recommended.</div>
            </div>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-800 shadow-sm">
            <div className="font-semibold text-gray-900">Pro tip</div>
            Use short, varied sentences for faster embedding. You can always edit after generation.
          </div>
        </div>
      </div>
    </div>
  </div>
);

const LoadingUI: FC<{ status: string; progress: number }> = ({
  status,
  progress,
}) => (
  <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center z-10 bg-black/50 backdrop-blur-sm">
    <div className="w-full max-w-md text-center p-4">
      <Logo className="w-24 mx-auto mb-6" />
      <h2 className="text-2xl font-bold mb-4">Initializing Brain...</h2>
      <div className="w-full bg-gray-700 rounded-full h-2.5 mb-2">
        <div
          className="bg-blue-500 h-2.5 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      <p className="text-gray-400 h-5">{status}</p>
    </div>
  </div>
);

export interface NeuronPoint {
  text: string;
  position: [number, number, number];
  embedding: number[];
}

export interface SearchResult extends NeuronPoint {
  similarity: number;
}

interface InteractiveSphereProps {
  point: NeuronPoint;
  color: string;
  similarity: number | null;
  onClick: (point: NeuronPoint) => void;
  isPreEmbedding: boolean;
  isLowQuality: boolean;
}

const InteractiveSphere: FC<InteractiveSphereProps> = ({
  point,
  color,
  similarity,
  onClick,
  isPreEmbedding,
  isLowQuality,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null!);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null!);
  const labelRef = useRef<HTMLDivElement>(null!);
  const { camera, invalidate } = useThree();
  const clockRef = useRef(0);
  const touchHideTimeout = useRef<number | null>(null);
  const [px, py] = point.position;

  useEffect(() => {
    return () => {
      if (touchHideTimeout.current !== null) {
        window.clearTimeout(touchHideTimeout.current);
      }
    };
  }, []);

  useFrame(() => {
    if (!meshRef.current || !materialRef.current) return;
    clockRef.current += 1;

    if (materialRef.current.opacity < 1) {
      materialRef.current.opacity = THREE.MathUtils.lerp(
        materialRef.current.opacity,
        1,
        0.05,
      );
    }

    const dist = meshRef.current.position.distanceTo(camera.position);
    const distanceScale = THREE.MathUtils.mapLinear(dist, 100, 25, 2.0, 1.0);
    const clampedDistanceScale = THREE.MathUtils.clamp(distanceScale, 1.0, 2.0);
    const hoverScale = isHovered ? 1.25 : 1.0;

    const sphereVisibilityScale =
      materialRef.current.opacity * clampedDistanceScale;
    const meshScale = sphereVisibilityScale * hoverScale;
    meshRef.current.scale.set(meshScale, meshScale, meshScale);

    if (labelRef.current) {
      labelRef.current.style.transform = `translateX(-50%) scale(${materialRef.current.opacity})`;
    }

    const baseIntensity = similarity !== null ? 1.2 : isPreEmbedding ? 0.55 : 0.35;
    const pulse =
      similarity !== null
        ? 0.4 * Math.sin(clockRef.current * 0.06 + px * 0.2 + py * 0.15)
        : isPreEmbedding
          ? 0.35 * Math.sin(clockRef.current * 0.09 + px * 0.27 + py * 0.21)
          : 0;
    materialRef.current.emissiveIntensity = baseIntensity + pulse;

    invalidate();
  });

  const labelText =
    similarity !== null
      ? `(${similarity.toFixed(2)}) ${point.text}`
      : point.text;
  const glowIntensity = similarity !== null ? 1.0 : 0.4;

  return (
    <group position={point.position}>
      <mesh
        ref={meshRef}
        onClick={() => onClick(point)}
        onPointerOver={(e) => {
          e.stopPropagation();
          setIsHovered(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setIsHovered(false);
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          if (touchHideTimeout.current !== null) {
            window.clearTimeout(touchHideTimeout.current);
          }
          setIsHovered(true);
          touchHideTimeout.current = window.setTimeout(() => {
            setIsHovered(false);
          }, 2000);
        }}
      >
        <sphereGeometry args={[0.25, isLowQuality ? 12 : 16, isLowQuality ? 12 : 16]} />
        <meshStandardMaterial
          ref={materialRef}
          color={color}
          roughness={0.5}
          emissive={color}
          emissiveIntensity={glowIntensity}
          transparent
          opacity={0}
        />
      </mesh>
      {isHovered && (
        <Html distanceFactor={12}>
          <div
            ref={labelRef}
            className="text-white bg-black/60 p-1.5 rounded-md text-sm whitespace-nowrap shadow-lg backdrop-blur-md"
            style={{
              transformOrigin: "center top",
              userSelect: "none",
            }}
          >
            <div>{labelText}</div>
          </div>
        </Html>
      )}
    </group>
  );
};

interface SceneProps {
  galaxyPoints: NeuronPoint[];
  searchResults: SearchResult[];
  onSphereClick: (point: NeuronPoint) => void;
  searchQuery: string;
  qualityLevel: QualityLevel;
}

interface LightningSegment {
  id: number;
  start: [number, number, number];
  end: [number, number, number];
  life: number;
}

const Scene: FC<SceneProps> = ({
  galaxyPoints,
  searchResults,
  onSphereClick,
  searchQuery,
  qualityLevel,
}) => {
  const assetBase = import.meta.env.BASE_URL || "/";
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const brainGroupRef = useRef<THREE.Group>(null);
  const cameraTargetPos = useRef(new THREE.Vector3());
  const controlsTargetLookAt = useRef(new THREE.Vector3());
  const shouldAnimate = useRef(false);
  const lightningId = useRef(0);
  const [lightningSegments, setLightningSegments] = useState<LightningSegment[]>([]);
  const { camera, invalidate } = useThree();

  const gltf = useGLTF(`${assetBase}brain_hologram.glb`);
  const brainObject = useMemo(() => gltf?.scene?.clone() ?? null, [gltf]);

  useEffect(() => {
    if (!brainObject) return;
    brainObject.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.material = new THREE.MeshStandardMaterial({
          color: "#5a3fbf",
          emissive: "#2a1766",
          emissiveIntensity: 0.3,
          transparent: true,
          opacity: 0.5,
          roughness: 1,
          side: THREE.DoubleSide,
        });
      }
    });
  }, [brainObject]);

  const brainPlacement = useMemo(() => {
    if (!brainObject) {
      return {
        scale: 1,
        offset: new THREE.Vector3(0, 0, 0),
        halfSize: new THREE.Vector3(55, 55, 55),
      };
    }
    const box = new THREE.Box3().setFromObject(brainObject);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = 100 / maxDim;
    const offset = center.multiplyScalar(-scale);
    const halfSize = size.clone().multiplyScalar(scale / 2);
    return { scale, offset, halfSize };
  }, [brainObject]);

  const brainWireframe = useMemo(() => {
    if (!brainObject) return null;
    let geometry: THREE.BufferGeometry | null = null;
    brainObject.traverse((child) => {
      if (!geometry && (child as THREE.Mesh).isMesh) {
        geometry = ((child as THREE.Mesh).geometry as THREE.BufferGeometry)
          .clone();
      }
    });
    return geometry ? new THREE.EdgesGeometry(geometry) : null;
  }, [brainObject]);

  const innerScale = 0.66;
  const innerOffset = useMemo(() => new THREE.Vector3(-2, 18, 0), []);
  const innerShapeAdjust = useMemo(
    () => new THREE.Vector3(1.30, 0.75, 1.2),
    [],
  );
  const innerAxes = useMemo(
    () =>
      brainPlacement.halfSize
        .clone()
        .multiplyScalar(innerScale)
        .multiply(innerShapeAdjust),
    [brainPlacement.halfSize, innerScale, innerShapeAdjust],
  );
  const innerAxesMargin = useMemo(
    () => innerAxes.clone().multiplyScalar(0.95),
    [innerAxes],
  );
  const innerCenter = useMemo(
    () => brainPlacement.offset.clone().negate().add(innerOffset),
    [brainPlacement.offset, innerOffset],
  );

  const innerEllipsoidGeometry = useMemo(
    () => new THREE.SphereGeometry(1, 48, 48),
    [],
  );

  const defaultView = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null);

  const computeDefaultView = useCallback(() => {
    if (!controlsRef.current) return null;

    const center = brainPlacement.offset.clone().negate();
    const maxDim =
      Math.max(
        brainPlacement.halfSize.x,
        brainPlacement.halfSize.y,
        brainPlacement.halfSize.z,
      ) * 2;
    const fov = (camera as THREE.PerspectiveCamera).fov * (Math.PI / 180);
    const baseDistance = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    const isMobile =
      typeof window !== "undefined" &&
      (window.matchMedia("(max-width: 768px)").matches ||
        /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent));
    // Keep a consistent view anchored to the brain model (not the points),
    // but stay relatively close so the brain fills the frame. Pull back a bit more on mobile.
    const distance = baseDistance * (isMobile ? 1.35 : 0.85);
    const pos = new THREE.Vector3(
      center.x + maxDim * 0.25,
      center.y + maxDim * 0.12,
      center.z + distance,
    );

    return { pos, target: center };
  }, [brainPlacement.halfSize.x, brainPlacement.halfSize.y, brainPlacement.halfSize.z, brainPlacement.offset, camera]);

  const placeholderPoints = useMemo(() => {
    if (!brainObject) return [];
    const isLowQuality = qualityLevel === "low";
    const count = isLowQuality ? 90 : 200;
    const pts: NeuronPoint[] = [];
    for (let i = 0; i < count; i++) {
      let v = new THREE.Vector3();
      do {
        v.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      } while (v.lengthSq() > 1);
      v.multiplyScalar(0.97);
      const mapped = new THREE.Vector3(
        v.x * innerAxesMargin.x + innerCenter.x,
        v.y * innerAxesMargin.y + innerCenter.y,
        v.z * innerAxesMargin.z + innerCenter.z,
      );
      pts.push({
        text: `Placeholder ${i + 1}`,
        position: [mapped.x, mapped.y, mapped.z],
        embedding: [],
      });
    }
    return pts;
  }, [brainObject, innerAxesMargin, innerCenter, qualityLevel]);

  const fitPoints = useMemo(() => {
    if (!brainObject) return galaxyPoints;
    if (galaxyPoints.length === 0) return galaxyPoints;

    // Get the original UMAP positions and normalize them
    const originalPositions = galaxyPoints.map((p) => new THREE.Vector3(...p.position));

    // Find bounds of original positions
    const box = new THREE.Box3().setFromPoints(originalPositions);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    return galaxyPoints.map((p, index) => {
      // Normalize the original UMAP position to -1 to 1 range
      const original = new THREE.Vector3(...p.position);
      const normalized = original.clone().sub(center).divideScalar(maxDim / 2);

      // Check if point is inside the unit sphere
      const length = normalized.length();

      // If outside unit sphere, project it inside with some margin
      if (length > 0.95) {
        normalized.normalize().multiplyScalar(0.95);
      }

      // Ensure minimum distance from center to avoid overlapping at origin
      if (length < 0.05) {
        // Use seeded random to push slightly outward
        const seededRandom = (seed: number) => {
          const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
          return x - Math.floor(x);
        };
        const pushDir = new THREE.Vector3(
          seededRandom(index * 3) * 2 - 1,
          seededRandom(index * 3 + 1) * 2 - 1,
          seededRandom(index * 3 + 2) * 2 - 1,
        ).normalize();
        normalized.copy(pushDir.multiplyScalar(0.05 + seededRandom(index * 7) * 0.1));
      }

      normalized.multiplyScalar(0.97);

      // Transform to ellipsoid space with custom shape
      const x = normalized.x * innerAxesMargin.x + innerCenter.x;
      const y = normalized.y * innerAxesMargin.y + innerCenter.y;
      const z = normalized.z * innerAxesMargin.z + innerCenter.z;

      return {
        ...p,
        position: [x, y, z] as [number, number, number],
      };
    });
  }, [galaxyPoints, innerAxesMargin, innerCenter, brainObject]);

  const frameBrain = useCallback(() => {
    if (!controlsRef.current) return;
    const view = computeDefaultView();
    if (!view) return;
    camera.position.copy(view.pos);
    controlsRef.current.target.copy(view.target);
    controlsRef.current.update();
    invalidate();
    defaultView.current = {
      pos: view.pos.clone(),
      target: view.target.clone(),
    };
  }, [computeDefaultView, camera, invalidate]);

  useEffect(() => {
    frameBrain();
  }, [frameBrain]);

  useEffect(() => {
    if (fitPoints.length === 0 || !controlsRef.current) return;
    const view = computeDefaultView();
    if (!view) return;
    camera.position.copy(view.pos);
    controlsRef.current.target.copy(view.target);
    controlsRef.current.update();
    invalidate();
    defaultView.current = {
      pos: view.pos.clone(),
      target: view.target.clone(),
    };
  }, [fitPoints.length, computeDefaultView, camera, invalidate]);

  useEffect(() => {
    if (!controlsRef.current) return;

    if (searchResults.length === 0 || !searchResults[0]?.text) {
      if (defaultView.current) {
        camera.position.copy(defaultView.current.pos);
        controlsRef.current.target.copy(defaultView.current.target);
        controlsRef.current.update();
        invalidate();
      }
      return;
    }

    const topResult = searchResults[0];

    const matchingFitPoint = fitPoints.find((fp) => fp.text === topResult.text);
    if (!matchingFitPoint) return;

    const topResultPos = new THREE.Vector3(...matchingFitPoint.position);
    const offsetDirection = new THREE.Vector3()
      .subVectors(camera.position, controlsRef.current.target)
      .normalize();
    const minFocusDist = 6;
    const maxFocusDist = 20;
    const similarity = THREE.MathUtils.clamp(topResult.similarity, 0, 1);
    const baseDist = THREE.MathUtils.mapLinear(
      similarity,
      0,
      1,
      maxFocusDist,
      minFocusDist,
    );
    const isMobile =
      typeof window !== "undefined" &&
      (window.matchMedia("(max-width: 768px)").matches ||
        /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent));
    const lengthFactor = THREE.MathUtils.clamp(
      (topResult.text.length - 40) / 80,
      0,
      1,
    );
    const distanceAdjust = isMobile ? 1 + 0.4 * lengthFactor : 1;
    const desiredDist = baseDist * distanceAdjust;
    const newOffset = offsetDirection.multiplyScalar(desiredDist);
    cameraTargetPos.current.copy(topResultPos).add(newOffset);
    controlsTargetLookAt.current.copy(topResultPos);
    shouldAnimate.current = true;
  }, [searchResults, camera, fitPoints, invalidate]);

  useEffect(() => {
    if (searchResults.length === 0 && defaultView.current && controlsRef.current) {
      shouldAnimate.current = false;
      controlsRef.current.enabled = true;
      camera.position.copy(defaultView.current.pos);
      controlsRef.current.target.copy(defaultView.current.target);
      controlsRef.current.update();
      invalidate();
    }
  }, [searchResults.length, frameBrain, camera, invalidate]);

  useEffect(() => {
    if (!searchQuery.trim()) {
      shouldAnimate.current = false;
      if (defaultView.current && controlsRef.current) {
        controlsRef.current.enabled = true;
        camera.position.copy(defaultView.current.pos);
        controlsRef.current.target.copy(defaultView.current.target);
        controlsRef.current.update();
        invalidate();
      } else {
        frameBrain();
      }
    }
  }, [searchQuery, frameBrain]);

  useFrame(() => {
    if (galaxyPoints.length === 0 && brainGroupRef.current) {
      brainGroupRef.current.rotation.y += 0.003;
      invalidate();
    }

    if (shouldAnimate.current && controlsRef.current) {
      controlsRef.current.enabled = false;
      const distToTarget = camera.position.distanceTo(cameraTargetPos.current);
      if (distToTarget > 0.01) {
        camera.position.lerp(cameraTargetPos.current, 0.08);
        controlsRef.current.target.lerp(controlsTargetLookAt.current, 0.08);
      } else {
        camera.position.copy(cameraTargetPos.current);
        controlsRef.current.target.copy(controlsTargetLookAt.current);
        shouldAnimate.current = false;
        controlsRef.current.enabled = true;
      }
      invalidate();
    }
  });

  const renderPoints = galaxyPoints.length > 0 ? fitPoints : placeholderPoints;
  const isPreEmbedding = galaxyPoints.length === 0;

  useEffect(() => {
    if (!isPreEmbedding) {
      setLightningSegments([]);
      return;
    }

    if (placeholderPoints.length === 0) return;

    const interval = window.setInterval(() => {
      setLightningSegments((prev) => {
        const alive = prev
          .map((seg) => ({ ...seg, life: seg.life - 0.24 }))
          .filter((seg) => seg.life > 0);
        const maxSegments = qualityLevel === "low" ? 3 : 6;
        while (alive.length < maxSegments) {
          const a = Math.floor(Math.random() * placeholderPoints.length);
          let b = Math.floor(Math.random() * placeholderPoints.length);
          if (a === b) {
            b = (b + 1) % placeholderPoints.length;
          }
          alive.push({
            id: lightningId.current++,
            start: placeholderPoints[a].position,
            end: placeholderPoints[b].position,
            life: 1.0,
          });
        }
        return alive;
      });
    }, 240);

    return () => window.clearInterval(interval);
  }, [isPreEmbedding, placeholderPoints, qualityLevel]);

  const { pointColors, similarityMap } = useMemo(() => {
    const idle = new THREE.Color("#34215f");
    const mid = new THREE.Color("#5a3fbf");
    const active = new THREE.Color("#9b7bff");
    const basePoints =
      galaxyPoints.length > 0 ? galaxyPoints : placeholderPoints;
    if (searchResults.length === 0) {
      if (galaxyPoints.length === 0) {
        const count = Math.max(placeholderPoints.length, 1);
        const spectrum = placeholderPoints.map((_, i) => {
          const t = count === 1 ? 0 : i / (count - 1);
          const r = 0.5 + 0.5 * Math.sin(2 * Math.PI * t);
          const g = 0.5 + 0.5 * Math.sin(2 * Math.PI * t + (2 * Math.PI) / 3);
          const b = 0.5 + 0.5 * Math.sin(2 * Math.PI * t + (4 * Math.PI) / 3);
          return `#${new THREE.Color(r, g, b).getHexString()}`;
        });
        return {
          pointColors: spectrum,
          similarityMap: new Map<string, number>(),
        };
      }
      return {
        pointColors: basePoints.map(() => "#5a3fbf"),
        similarityMap: new Map<string, number>(),
      };
    }
    const simMap = new Map(searchResults.map((r) => [r.text, r.similarity]));
    const colors = basePoints.map((point) => {
      const similarity = simMap.get(point.text);
      if (similarity === undefined) {
        return "#5a3fbf";
      }
      const color = new THREE.Color();
      const clamped = THREE.MathUtils.clamp(similarity, 0, 1);
      color.lerpColors(idle, mid, Math.min(clamped * 2, 1));
      color.lerpColors(color, active, Math.max(0, clamped - 0.5) * 2);
      return `#${color.getHexString()}`;
    });
    return { pointColors: colors, similarityMap: simMap };
  }, [galaxyPoints, searchResults, placeholderPoints]);

  return (
    <>
      <fog attach="fog" args={["#e8ecf2", 60, 220]} />
      <ambientLight intensity={0.6} />
      <hemisphereLight intensity={0.35} args={["#ffffff", "#888888", 1]} />
      <pointLight position={[0, 40, 20]} intensity={1.4} />
      <pointLight position={[0, -30, -10]} intensity={0.6} color="#7cf0ff" />
      <OrbitControls ref={controlsRef} makeDefault enableZoom enablePan />
      {brainObject && (
        <group
          ref={brainGroupRef}
          scale={[
            brainPlacement.scale,
            brainPlacement.scale,
            brainPlacement.scale,
          ]}
          position={[
            brainPlacement.offset.x,
            brainPlacement.offset.y,
            brainPlacement.offset.z,
          ]}
        >
          <primitive object={brainObject} />
          {brainWireframe && (
            <lineSegments>
              <primitive object={brainWireframe} attach="geometry" />
              <lineBasicMaterial
                color="#a78bfa"
                transparent
                opacity={0.35}
              />
            </lineSegments>
          )}
        </group>
      )}
      <mesh
        geometry={innerEllipsoidGeometry}
        scale={[
          innerAxes.x,
          innerAxes.y,
          innerAxes.z,
        ]}
        position={[
          innerCenter.x,
          innerCenter.y,
          innerCenter.z,
        ]}
        visible={false}
      >
        <meshBasicMaterial color="#000000" opacity={0} transparent />
      </mesh>

      {renderPoints.map((point, i) => (
        <InteractiveSphere
          key={point.text + i}
          point={point}
          color={pointColors[i]}
          similarity={similarityMap.get(point.text) ?? null}
          onClick={onSphereClick}
          isPreEmbedding={isPreEmbedding}
          isLowQuality={qualityLevel === "low"}
        />
      ))}
      {isPreEmbedding && lightningSegments.length > 0 && (
        <group>
          {lightningSegments.map((seg) => (
            <line key={seg.id}>
              <bufferGeometry
                attach="geometry"
                onUpdate={(geo) =>
                  geo.setFromPoints([
                    new THREE.Vector3(...seg.start),
                    new THREE.Vector3(...seg.end),
                  ])
                }
              />
              <lineBasicMaterial
                color="#7cf8ff"
                transparent
                opacity={Math.max(0, Math.min(1, seg.life))}
                linewidth={1}
              />
            </line>
          ))}
        </group>
      )}
    </>
  );
};

export default function App() {
  const {
    device,
    loadModel,
    isLoading,
    isReady,
    progress,
    status,
    error,
    embed,
  } = useModel();

  const [quality, setQuality] = useState<QualityLevel>(() => detectQualityPreset());
  const qualitySettings = QUALITY_PRESETS[quality];

  const [textInput, setTextInput] = useState<string>("");
  const [galaxyPoints, setGalaxyPoints] = useState<NeuronPoint[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [isTextareaExpanded, setIsTextareaExpanded] = useState<boolean>(false);
  const [isMusicMuted, setIsMusicMuted] = useState<boolean>(false);
  const lastQueryEmbedding = useRef<number[] | null>(null);
  const [generationStatus, setGenerationStatus] = useState("");

  const isSearching = useRef(false);
  const pendingQuery = useRef<string | null>(null);

  const setDefaultSentences = () => {
    let sentences = DEFAULT_SENTENCES;
    if (device === "wasm") {
      // Use fewer examples for demonstration purposes (it's slower)
      sentences = sentences.filter((_, i) => i % 2 === 0);
    }
    setTextInput(sentences.join("\n"));
  };

  useEffect(() => {
    setDefaultSentences();
  }, [device]);

  const handleGenerateGalaxy = async () => {
    if (!isReady || !textInput.trim()) {
      alert("Model not ready or no text provided.");
      return;
    }
    setIsGenerating(true);
    setSearchResults([]);
    setSearchQuery("");
    lastQueryEmbedding.current = null;
    setGenerationStatus("Generating brain...");
    const sentences = textInput
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
      .sort((x) => x.length);
    if (sentences.length < 3) {
      alert(
        "Please provide at least 3 sentences for UMAP to work effectively.",
      );
      setIsGenerating(false);
      return;
    }

    const batch_size = device === "webgpu" ? 4 : 1;
    try {
      const embeddings: number[][] = [];
      setGenerationStatus(`Embedding... (0%)`);
      for (let i = 0; i < sentences.length; i += batch_size) {
        const batch = sentences.slice(i, i + batch_size);
        const progress = ((i + batch.length) / sentences.length) * 100;

        const batchEmbeddings = await embed(batch, {
          padding: true,
          truncation: true,
          max_length: 256,
        });
        embeddings.push(...batchEmbeddings);
        setGenerationStatus(`Embedding... (${progress.toFixed(0)}%)`);
      }
      setGenerationStatus("Running UMAP to create 3D projection...");
      const nNeighbors = Math.max(2, Math.min(sentences.length - 1, 15));
      const umap = new UMAP({ nComponents: 3, nNeighbors, minDist: 0.1 });
      const coords3D: number[][] = umap.fit(embeddings);
      const rawPoints = coords3D.map((p) => new THREE.Vector3(...p));
      const box = new THREE.Box3().setFromPoints(rawPoints);
      const center = box.getCenter(new THREE.Vector3());
      const centeredPoints = rawPoints.map((p) => p.sub(center));
      let maxDist = 0;
      for (const p of centeredPoints) {
        maxDist = Math.max(maxDist, p.length());
      }
      const scaleFactor = 50;
      const finalPoints = centeredPoints.map((p) => {
        const normalized = maxDist > 0 ? p.divideScalar(maxDist) : p;
        const brainPos = warpToBrainShape(normalized);
        return brainPos.multiplyScalar(scaleFactor);
      });
      const positions = finalPoints.map((p) => p.toArray()) as [
        number,
        number,
        number,
      ][];
      const newPoints: NeuronPoint[] = sentences.map((text, i) => ({
        text,
        position: positions[i],
        embedding: embeddings[i],
      }));
      setGalaxyPoints(newPoints);
      setGenerationStatus(
        `Brain generated with ${newPoints.length} neuron sparks. Ready to explore!`,
      );
      setIsSidebarOpen(false);
    } catch (e) {
      setGenerationStatus("An error occurred during generation.");
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    pendingQuery.current = searchQuery;

    const processQueue = async () => {
      if (isSearching.current || pendingQuery.current === null) {
        return;
      }

      isSearching.current = true;
      const queryToRun = pendingQuery.current;
      pendingQuery.current = null;

      if (!queryToRun.trim() || !isReady || galaxyPoints.length === 0) {
        setSearchResults([]);
        lastQueryEmbedding.current = null;
        isSearching.current = false;
        if (pendingQuery.current !== null) processQueue();
        return;
      }

      try {
        const [queryEmbedding] = await embed([queryToRun], {
          padding: true,
          truncation: true,
          max_length: 256,
        });
        lastQueryEmbedding.current = queryEmbedding;
        const results: SearchResult[] = galaxyPoints
          .map((point) => ({
            ...point,
            similarity: cos_sim(queryEmbedding, point.embedding),
          }))
          .sort((a, b) => b.similarity - a.similarity);
        setSearchResults(results);
      } catch (e) {
        /* ignore search errors to avoid noisy logs */
      } finally {
        isSearching.current = false;
        if (pendingQuery.current !== null) {
          processQueue();
        }
      }
    };

    processQueue();
  }, [searchQuery, galaxyPoints, isReady, embed]);

  const handlePointFocus = (point: NeuronPoint | SearchResult) => {
    let similarity = (point as SearchResult).similarity;
    if (similarity === undefined) {
      if (lastQueryEmbedding.current) {
        similarity = cos_sim(lastQueryEmbedding.current, point.embedding);
      } else {
        return;
      }
    }
    const focusedResult: SearchResult = { ...point, similarity };
    const newResults = [
      focusedResult,
      ...searchResults.filter((r) => r.text !== point.text),
    ];
    setSearchResults(newResults);
  };

  const renderBrainCanvas = () => (
    <Canvas
      camera={{ position: [0, 0, 25], fov: 45 }}
      dpr={qualitySettings.dpr}
      gl={{
        antialias: qualitySettings.antialias,
        powerPreference: qualitySettings.powerPreference,
      }}
    >
      <color attach="background" args={["#f2f2f2"]} />
      <Suspense
        fallback={
          <Html center>
            <div className="text-gray-900">Loading 3D Scene...</div>
          </Html>
        }
      >
        <Scene
          galaxyPoints={galaxyPoints}
          searchResults={searchResults}
          onSphereClick={handlePointFocus}
          searchQuery={searchQuery}
          qualityLevel={quality}
        />
        {qualitySettings.enableBloom && (
          <EffectComposer enableNormalPass={false}>
            <Bloom
              luminanceThreshold={0.1}
              luminanceSmoothing={0.9}
              height={qualitySettings.bloomHeight}
              intensity={qualitySettings.bloomIntensity}
            />
          </EffectComposer>
        )}
      </Suspense>
    </Canvas>
  );

  if (!isReady) {
    return (
      <div className="h-screen w-screen bg-[#f2f2f2] text-gray-900 relative">
        <BackgroundMusic muted={isMusicMuted} />
          <div className="absolute top-0 left-0 w-full h-full z-0">
            {renderBrainCanvas()}
          </div>
        {!isLoading && <MainMenuUI onLoadModel={loadModel} />}
        {isLoading && <LoadingUI status={status} progress={progress} />}
        {error && (
          <div className="absolute bottom-4 left-4 bg-red-500/50 text-white p-4 rounded-lg">
            <p>Error: {error}</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-[#f2f2f2] text-gray-900 relative">
      <BackgroundMusic muted={isMusicMuted} />
      <div className="absolute top-0 left-0 w-full h-full z-0">
        {renderBrainCanvas()}
      </div>
            <div className="absolute top-0 left-0 w-full h-full pointer-events-none z-10">
        <div
          className={`absolute top-0 left-0 h-full bg-white/90 backdrop-blur-xl border-r border-gray-300 transition-transform duration-300 ease-in-out ${
            isSidebarOpen ? "translate-x-0" : "-translate-x-full"
          } pointer-events-auto text-gray-900 shadow-2xl`}
          style={{ width: "min(420px, 92vw)" }}
        >
          <div className="flex flex-col h-full p-6 gap-4">
            <div
              className={`flex flex-col transition-all duration-300 ease-in-out ${
                isTextareaExpanded ? "flex-grow-[10]" : "flex-grow-[2]"
              }`}
            >
              <div className="flex gap-2 items-center mb-2">
                <Logo className="w-12 ml-[-6px]" />
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 leading-tight">ThoughtNebula</h1>
                  <p className="text-sm text-gray-600">
                    Turn your text into glowing neurons inside a brain.
                  </p>
                </div>
              </div>
              <div className="flex items-center justify-between mb-3 gap-2">
                <label className="font-semibold text-gray-700">Performance</label>
                <select
                  value={quality}
                  onChange={(e) =>
                    setQuality(e.target.value as QualityLevel)
                  }
                  className="bg-white text-sm rounded-md px-2 py-1 border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:outline-none text-gray-900"
                >
                  <option value="low">Low · battery saver</option>
                  <option value="medium">Medium · balanced</option>
                  <option value="high">High · max fidelity</option>
                </select>
              </div>
              <div className="flex justify-between items-center mb-1">
                <label
                  htmlFor="text-input"
                  className="font-semibold text-gray-700"
                >
                  Your Dataset
                </label>
                <div className="flex items-center gap-3">
                  <button
                    onClick={setDefaultSentences}
                    className="text-sm font-medium text-blue-600 hover:text-blue-500 transition-colors"
                  >
                    Try Example
                  </button>
                  <button
                    onClick={() => setIsMusicMuted((prev) => !prev)}
                    aria-pressed={isMusicMuted}
                    className="text-sm font-medium text-blue-600 hover:text-blue-500 transition-colors"
                  >
                    {isMusicMuted ? "Unmute Music" : "Mute Music"}
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-600 mb-2">
                Enter full sentences, one per line. Short, natural sentences embed best.
              </p>
              <textarea
                id="text-input"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onFocus={() => setIsTextareaExpanded(true)}
                onBlur={() => setIsTextareaExpanded(false)}
                className="flex-grow bg-white border border-gray-300 rounded-md p-3 text-sm text-gray-900 resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none whitespace-pre-wrap overflow-auto transition-all duration-300 ease-in-out shadow-inner"
                placeholder="Enter sentences here, one per line."
              />
              <button
                onClick={handleGenerateGalaxy}
                disabled={isGenerating || !isReady}
                className="mt-4 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-xl transition-colors shadow-md"
              >
                {isGenerating ? generationStatus : "Generate Brain"}
              </button>
              <p className="text-center text-sm mt-2 text-gray-600 h-5">
                {!isGenerating ? generationStatus : ""}
              </p>
            </div>
            {galaxyPoints.length > 0 && (
              <div
                className={`mt-4 flex flex-col min-h-0 transition-all duration-300 ease-in-out ${
                  isTextareaExpanded
                    ? "flex-grow-[1] opacity-50"
                    : "flex-grow-[3] opacity-100"
                }`}
              >
                <h2 className="font-semibold mb-2 text-gray-800">Search Results</h2>
                <div className="overflow-y-auto pr-2">
                  {searchResults.length === 0 && (
                    <p className="text-sm text-gray-600">
                      Search to see results.
                    </p>
                  )}
                  {searchResults.map((result, i) => (
                    <div
                      key={result.text + i}
                      onClick={() => handlePointFocus(result)}
                      className={`p-2 mb-1 rounded-md cursor-pointer transition-colors ${
                        i === 0
                          ? "bg-blue-50 border border-blue-200"
                          : "bg-gray-100 hover:bg-gray-200 border border-gray-200"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <p className="font-semibold text-sm truncate pr-2 text-gray-900">
                          {result.text}
                        </p>
                        <span className="text-xs font-mono bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">
                          {result.similarity.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="absolute top-6 bg-black/30 backdrop-blur-lg p-2 rounded-full transition-all duration-300 ease-in-out pointer-events-auto"
          style={{
            left: isSidebarOpen ? "min(400px, 90vw)" : "0",
            transform: "translateX(1.5rem)",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={`h-6 w-6 text-white transition-transform duration-300 ${isSidebarOpen ? "rotate-180" : "rotate-0"}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
        {galaxyPoints.length > 0 && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 pointer-events-auto">
            <div className="relative">
              <input
                type="text"
                placeholder="Type words to search for similar neurons..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-black/30 backdrop-blur-lg border border-white/10 rounded-full py-3 px-8 text-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
