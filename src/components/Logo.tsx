import type { ImgHTMLAttributes, FC } from "react";
import logoSrc from "/logo.png";

const Logo: FC<ImgHTMLAttributes<HTMLImageElement>> = (props) => {
  return <img src={logoSrc} alt="ThoughtNebula logo" {...props} />;
};

export default Logo;
