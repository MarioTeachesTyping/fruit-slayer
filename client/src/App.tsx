import { useState } from "react";

export default function App() {
  const [playing, setPlaying] = useState(false);

  const onPlay = async () => {
    try {
      await fetch("/start", { method: "POST" }); // Vite proxy -> :5000
    } catch (e) {
      // non-fatal in dev; still switch to playing view
      console.warn("Failed to start backend:", e);
    }
    setPlaying(true);
  };

  if (playing) {
  return (
    <div
      className="grid min-h-screen place-items-center text-white bg-cover bg-center bg-no-repeat"
      style={{ backgroundImage: `url(/play-bg.png)` }}
    >
      <div className="text-center space-y-4 bg-black/40 p-6 rounded-xl">
        <p className="text-xl">Game window should be open.</p>
        <p className="opacity-70">
          If you don't see it, check behind other windows.
        </p>
        <button
          onClick={() => setPlaying(false)}
          className="rounded-2xl bg-sky-600 px-6 py-2"
        >
          Back
        </button>
      </div>
    </div>
  );
}

  return (
    <div
      className="min-h-screen w-full bg-cover bg-center bg-no-repeat relative"
      style={{ backgroundImage: `url(/title-screen.png)` }}
    >
      <div className="absolute inset-0 bg-black/30" />

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center p-6">
        <div className="relative flex items-end gap-6">
          <img
            src="/fruit-logo.png"
            alt="Fruit"
            className="w-56 md:w-72 lg:w-80 -translate-x-4 drop-shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
          />
          <img
            src="/slayer-logo.png"
            alt="Slayer"
            className="w-56 md:w-72 lg:w-112 translate-x-4 -mt-4 opacity-95 drop-shadow-[0_8px_24px_rgba(0,0,0,0.35)]"
          />
        </div>

        <button
          onClick={onPlay}
          className="mt-10 rounded-2xl bg-sky-600 px-10 py-3 text-lg font-semibold text-white transition-transform hover:scale-[1.02] hover:bg-sky-500 focus:outline-none"
        >
          Play
        </button>
      </div>
    </div>
  );
}