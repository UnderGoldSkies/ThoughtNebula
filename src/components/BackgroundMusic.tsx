import { useEffect, useRef } from "react";

interface BackgroundMusicProps {
  muted: boolean;
}

const BackgroundMusic = ({ muted }: BackgroundMusicProps) => {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    const isMobile =
      typeof window !== "undefined" &&
      (window.matchMedia("(max-width: 768px)").matches ||
        /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent));
    const audioVolume = isMobile ? 0.05 : 0.14;
    if (!audio) return;

    const playAudio = () => {
      audio.volume = audioVolume;
      audio.muted = muted;
      audio.play().catch((e) => console.error("Could not play audio on click.", e));
    };

    audio.volume = audioVolume;
    audio.muted = muted;
    const playPromise = audio.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        document.addEventListener("click", playAudio, { once: true });
      });
    }

    return () => {
      document.removeEventListener("click", playAudio);
    };
  }, [muted]);

  return (
    <audio
      ref={audioRef}
      src={`${import.meta.env.BASE_URL}music.mp3`}
      loop
    />
  );
};

export default BackgroundMusic;
