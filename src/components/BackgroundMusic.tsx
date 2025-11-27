import { useEffect, useRef } from "react";

const BackgroundMusic = () => {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    const audiovolume = 0.3;
    if (audio) {
      audio.volume = audiovolume;
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          const playOnClick = () => {
            audio.volume = audiovolume;
            audio
              .play()
              .catch((e) => console.error("Could not play audio on click.", e));
            document.removeEventListener("click", playOnClick);
          };
          document.addEventListener("click", playOnClick);
        });
      }
    }
  }, []);

  return <audio ref={audioRef} src="/music.mp3" loop />;
};

export default BackgroundMusic;
