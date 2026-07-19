import { useTranslation } from "react-i18next";

export default function LanguageToggle() {
  const { i18n } = useTranslation();
  const current = i18n.language.startsWith("zh") ? "zh" : "en";

  const switchTo = (lang: "zh" | "en") => {
    void i18n.changeLanguage(lang);
  };

  return (
    <div className="inline-flex items-center rounded-full border border-ob-border bg-ob-surface/70 p-0.5 text-xs backdrop-blur">
      {(["zh", "en"] as const).map((lang) => (
        <button
          key={lang}
          type="button"
          onClick={() => switchTo(lang)}
          className={`rounded-full px-3 py-1 transition-colors ${
            current === lang
              ? "bg-nebula text-white shadow-glow"
              : "text-ob-muted hover:text-ob-text"
          }`}
        >
          {lang === "zh" ? "中文" : "EN"}
        </button>
      ))}
    </div>
  );
}
