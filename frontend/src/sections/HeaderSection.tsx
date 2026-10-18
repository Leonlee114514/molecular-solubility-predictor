import { useTranslation } from "react-i18next";

export default function HeaderSection() {
  const { t } = useTranslation();

  return (
    <header className="flex flex-col items-center gap-3 px-2 pb-2 text-center">
      <p className="text-xs uppercase tracking-[0.18em] text-ob-muted">
        {t("header.tagline")}
      </p>
      <h1 className="gradient-title text-4xl font-bold leading-tight md:text-5xl">
        {t("header.title")}
      </h1>
      <p className="max-w-xl text-base text-ob-muted">
        {t("header.subtitle")}
      </p>
    </header>
  );
}
